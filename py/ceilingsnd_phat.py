import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import detrend
from scipy.io import wavfile
from numpy.lib.stride_tricks import as_strided

# Configure Chinese-capable fonts for plot text rendering.
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# Parameter configuration
# ==========================================
FFT_SIZE_OPTIONS = [8, 16, 32, 64, 128, 256, 512]
FFT_ADVANCE_STEP_OPTIONS = [2, 4, 8, 16]
MAX_SWEEP_ROUNDS = 5
RECORD_DURATION_SEC = 0.5
SPEED_OF_SOUND = 343.0
INTERP = 16
HF_MIN_FREQ_HZ = 400.0
#HF_MIN_FREQ_HZ = 400.0
HF_ENERGY_RATIO_MIN = 0.05
#HF_ENERGY_RATIO_MIN = 0.005

# v1: 6000.0, 0.3

def normalize_audio(audio):
    """Normalize to float32 and remove linear trend per channel."""
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32) / 32768.0
    return detrend(audio, axis=0, type='linear')

def frame_fft_array(sig, refsig, fft_size=256, advance_step=2):
    """
    Slice with sliding windows and compute FFT for each frame.
    Fix note: use correct as_strided stride calculation.
    """
    sig = np.ascontiguousarray(sig)
    refsig = np.ascontiguousarray(refsig)

    min_len = min(len(sig), len(refsig))
    if min_len < fft_size:
        raise ValueError(f"Not enough samples for one frame: min_len={min_len}, FFT_SIZE={fft_size}")
    
    # Number of valid frames for the current window and hop.
    frame_count = (min_len - fft_size) // advance_step + 1
    if frame_count <= 0:
        raise ValueError(f"advance_step={advance_step} is too large; no valid frames can be generated")
    
    # --- Fix: correct stride calculation ---
    # strides = (bytes per hop step, bytes per single sample)
    sig_stride = sig.strides[0] if hasattr(sig, 'strides') else 4 # Assume float32 (4 bytes)
    ref_stride = refsig.strides[0] if hasattr(refsig, 'strides') else 4
    
    shape_sig = (frame_count, fft_size)
    strides_sig = (sig_stride * advance_step, sig_stride)
    
    shape_ref = (frame_count, fft_size)
    strides_ref = (ref_stride * advance_step, ref_stride)
    
    try:
        sig_frames = as_strided(sig, shape=shape_sig, strides=strides_sig, writeable=False)
        ref_frames = as_strided(refsig, shape=shape_ref, strides=strides_ref, writeable=False)
    except ValueError as e:
        raise RuntimeError(f"Failed to create strided memory view: {e}") from e

    # Batch FFT across all frames.
    sig_fft = np.fft.rfft(sig_frames, n=fft_size, axis=1)
    ref_fft = np.fft.rfft(ref_frames, n=fft_size, axis=1)
    
    # Start sample index for each frame in the original signal.
    start_indices = np.arange(0, frame_count * advance_step, advance_step)
    
    return sig_fft, ref_fft, frame_count, start_indices

def frequency_distance_match(sig_fft, ref_fft, top_n=5):
    """
    Coarse matching based on cosine similarity.
    """
    sig_mag = np.abs(sig_fft)
    ref_mag = np.abs(ref_fft)
    
    # Cosine similarity per frame.
    dot_product = np.sum(sig_mag * ref_mag, axis=1)
    norm_product = np.linalg.norm(sig_mag, axis=1) * np.linalg.norm(ref_mag, axis=1)
    similarity = dot_product / (norm_product + 1e-15)
    
    # Indices of top-N most similar frames.
    top_n = max(1, min(top_n, similarity.shape[0]))
    best_indices = np.argsort(similarity)[-top_n:][::-1]
    return best_indices, similarity

def gcc_phat(sig, refsig, fs=1, max_tau=None, interp=INTERP, fft_size=256, advance_step=2):
    """
    Multi-frame fused GCC-PHAT.
    """
    try:
        sig_fft_frames, ref_fft_frames, frame_count, start_indices = frame_fft_array(
            sig, refsig, fft_size=fft_size, advance_step=advance_step
        )
    except ValueError as e:
        # If framing fails (e.g., hop is too large), re-raise directly.
        raise e

    # Get the indices of top-5 matched frames.
    best_indices, frame_similarity = frequency_distance_match(sig_fft_frames, ref_fft_frames, top_n=5)
    estimated_taus = []
    final_cc = None # For plotting: correlation curve from the best frame.

    for idx in best_indices:
        cross_spec = sig_fft_frames[idx] * np.conj(ref_fft_frames[idx])
        # PHAT weighting: keep phase information only.
        r_phat = cross_spec / (np.abs(cross_spec) + 1e-15)
        
        n = fft_size
        cc = np.fft.irfft(r_phat, n=(interp * n))
        
        max_shift = int(interp * n / 2)
        if max_tau is not None:
            max_shift = min(int(interp * fs * max_tau), max_shift)
            
        cc_clipped = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))
        
        # Locate peak index.
        shift = np.argmax(np.abs(cc_clipped)) - max_shift
        
        # Parabolic interpolation for sub-sample precision.
        peak_idx = int(shift + max_shift)
        if 0 < peak_idx < len(cc_clipped) - 1:
            alpha = np.abs(cc_clipped[peak_idx - 1])
            beta = np.abs(cc_clipped[peak_idx])
            gamma = np.abs(cc_clipped[peak_idx + 1])
            # Offset of parabola vertex around the peak.
            p = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma + 1e-15)
            shift += p
            
        tau = shift / float(interp * fs)
        estimated_taus.append(tau)

        # Save first-best frame's cross-correlation for plotting.
        if final_cc is None:
            final_cc = cc_clipped

    # Use median tau to suppress single-frame outliers.
    final_tau = float(np.median(estimated_taus))
    
    freq_axis = np.fft.rfftfreq(fft_size, d=1.0 / fs)
    # Keep compatibility with existing print logic: return primary best index.
    best_primary_idx = int(best_indices[0])
    peak_abs = float(np.max(np.abs(final_cc))) if final_cc is not None else 0.0
    
    best_sig_mag = np.abs(sig_fft_frames[best_primary_idx])
    best_ref_mag = np.abs(ref_fft_frames[best_primary_idx])
    # Frequency match score: 1 means close amplitude match, 0 means large mismatch.
    freq_match_scores = 1.0 - np.abs(best_sig_mag - best_ref_mag) / (best_sig_mag + best_ref_mag + 1e-15)
    freq_match_scores = np.clip(freq_match_scores, 0.0, 1.0)
    best_frame_similarity = float(frame_similarity[best_primary_idx])

    return (
        final_tau,
        final_cc,
        None,
        freq_axis,
        sig_fft_frames,
        ref_fft_frames,
        None,
        best_primary_idx,
        frame_count,
        start_indices,
        peak_abs,
        freq_match_scores,
        best_frame_similarity,
    )

def analyze_direction(tau, threshold=50e-6):
    """Infer left/right direction from TDOA."""
    if tau < -threshold:
        return "Sound reaches CH2 first -> source is closer to CH2 side"
    if tau > threshold:
        return "Sound reaches CH1 first -> source is closer to CH1 side"
    return "Both channels arrive almost simultaneously -> source is centered/far-field"

def record_audio_block(fs, duration_sec, device_index):
    """Record one dual-channel audio block with fixed duration."""
    audio = sd.rec(int(duration_sec * fs), samplerate=fs, channels=2, device=device_index, dtype='float32')
    sd.wait()
    return normalize_audio(audio)

def has_high_frequency_content(audio_block, fs, min_freq_hz=HF_MIN_FREQ_HZ, min_ratio=HF_ENERGY_RATIO_MIN):
    """Return True when spectral energy above min_freq_hz is significant enough."""
    if audio_block.ndim == 2:
        mono = np.mean(audio_block, axis=1)
    else:
        mono = audio_block

    if mono.size < 8:
        return False, 0.0

    # Window before FFT to reduce spectral leakage near the high-frequency band edge.
    windowed = mono * np.hanning(mono.size)
    spectrum = np.fft.rfft(windowed)
    power = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(mono.size, d=1.0 / fs)

    high_mask = freqs >= min_freq_hz
    if not np.any(high_mask):
        return False, 0.0

    total_energy = float(np.sum(power) + 1e-15)
    high_energy = float(np.sum(power[high_mask]))
    high_ratio = high_energy / total_energy
    return high_ratio >= min_ratio, high_ratio

def play_alert_sound(wav_path):
    """Play alert sound synchronously so microphone detection is paused during playback."""
    if not os.path.exists(wav_path):
        print(f"Positive > Negative, but file not found: {wav_path}")
        return

    try:
        wav_fs, wav_data = wavfile.read(wav_path)
        print(f"Positive magnitude > negative magnitude, playing alert: {wav_path}")
        sd.play(wav_data, wav_fs)
        sd.wait()
    except Exception as exc:
        print(f"Failed to play 13360.wav: {exc}")

def run_parameter_sweep(ch1, ch2, fs):
    """
    Sweep FFT_SIZE_OPTIONS × FFT_ADVANCE_STEP_OPTIONS.
    Logic: outer loop over FFT_SIZE, inner loop over FFT_ADVANCE_STEP.
    That is: fix one SIZE, iterate all STEP values, then move to next SIZE.
    """
    results = []
    total_combinations = len(FFT_SIZE_OPTIONS) * len(FFT_ADVANCE_STEP_OPTIONS)
    target_rounds = min(MAX_SWEEP_ROUNDS, total_combinations)
    current_round = 0

    # --- Core logic: nested loops ---
    # Outer loop: FFT size.
    for fft_size in FFT_SIZE_OPTIONS:
        # Inner loop: hop size.
        for advance_step in FFT_ADVANCE_STEP_OPTIONS:
            if current_round >= target_rounds:
                break

            current_round += 1
            print("=" * 40)
            print(f"Parameter sweep round: {current_round} / {target_rounds}")
            print(f"Current combination -> FFT_SIZE: {fft_size}, FFT_ADVANCE_STEP: {advance_step}")
            print("-" * 40)

            try:
                # Execute full sliding-window FFT/matching pipeline.
                (
                    tau,
                    cc,
                    _,
                    freq_axis,
                    sig_fft_frames,
                    ref_fft_frames,
                    _,
                    best_idx,
                    frame_count,
                    start_indices,
                    peak_abs,
                    freq_match_scores,
                    best_frame_similarity,
                ) = gcc_phat(
                    ch1, ch2, fs=fs, fft_size=fft_size, advance_step=advance_step
                )
                
                distance_diff_cm = tau * SPEED_OF_SOUND * 100
                direction_msg = analyze_direction(tau)
                score = peak_abs
                status = "SUCCESS"

                print(f"Status: {status}")
                print(f"TDOA: {tau * 1e6:.2f} us")
                print(f"Distance difference: {distance_diff_cm:.2f} cm")
                print(f"Direction: {direction_msg}")
                print(f"Peak strength score: {score:.6f}")
                print(f"Best-frame cosine similarity: {best_frame_similarity:.6f}")
                print(f"Framing stats: {frame_count} frames, {len(freq_axis)} frequency bins")
                # Count non-negative as positive so positive+negative equals total rounds.
                round_sign = "positive" if tau >= 0 else "negative"
                print(f"Round overall TDOA sign: {round_sign} (from final tau)")
                print("Frequency (Hz) array:")
                print(np.array2string(freq_axis, precision=1, separator=','))
                print("Per-frequency matching scores (0~1):")
                print(np.array2string(freq_match_scores, precision=4, separator=','))

                results.append({
                    "status": status,
                    "fft_size": fft_size,
                    "advance_step": advance_step,
                    "tau": float(tau),
                    "distance_diff_cm": float(distance_diff_cm),
                    "direction": direction_msg,
                    "score": float(score),
                    "frame_count": int(frame_count),
                    "best_idx": int(best_idx),
                    "window_start": int(start_indices[best_idx]),
                    "cc": cc.copy() if hasattr(cc, 'copy') else cc, # Store cross-correlation for plotting.
                    "best_frame_similarity": best_frame_similarity,
                    "freq_axis": freq_axis.copy(),
                    "freq_match_scores": freq_match_scores.copy(),
                })

            except Exception as exc:
                status = "FAILED"
                error_msg = str(exc)
                print(f"Status: {status}")
                print(f"Error details: {error_msg}")
                results.append({
                    "status": status,
                    "fft_size": fft_size,
                    "advance_step": advance_step,
                    "error": error_msg,
                    "score": -np.inf,
                    "tau": None
                })

                # --- Key interaction: wait for keyboard input ---
                # Pauses here until Enter is pressed, then continues to next combination.
            #input(f"[Waiting...] Finished combination (SIZE={fft_size}, STEP={advance_step}). Review the result and press <Enter> to continue...")

        if current_round >= target_rounds:
            break

    # ==========================================
            # End of sweep: generate final summary.
    # ==========================================
    print("\n" + "="*50)
    print("All parameter combinations have been scanned")
    print("="*50)
    
    ok_results = [r for r in results if r["status"] == "SUCCESS"]
    print(f"Round summary: target {target_rounds} | success {len(ok_results)} | failed {len(results) - len(ok_results)}")

    if not ok_results:
        print("Warning: No successful combinations; cannot produce a best-result summary.")
        return None, results

    # Sort by score.
    ranked = sorted(ok_results, key=lambda r: r["score"], reverse=True)
    print("\n--- Leaderboard (Top 5 by peak score) ---")
    for i, res in enumerate(ranked[:5], 1):
        print(f"{i}. SIZE={res['fft_size']}, STEP={res['advance_step']} | "
              f"score={res['score']:.4f} | TDOA={res['tau']*1e6:.2f}us | distance={res['distance_diff_cm']:.2f}cm")

    # Return the top-scored result.
    best_result = ranked[0]

    # Sum positive and negative TDOA values separately, then compare absolute magnitudes.
    positive_tau_sum = float(sum(r["tau"] for r in ok_results if r["tau"] > 0.0))
    negative_tau_sum = float(sum(r["tau"] for r in ok_results if r["tau"] < 0.0))
    positive_abs = abs(positive_tau_sum)
    negative_abs = abs(negative_tau_sum)

    if positive_abs > negative_abs:
        dominant_side = "positive"
    elif negative_abs > positive_abs:
        dominant_side = "negative"
    else:
        dominant_side = "tie"

    print("\n--- Overall TDOA Sum Comparison ---")
    print(f"Positive TDOA sum: {positive_tau_sum:+.8e} s | abs={positive_abs:.8e}")
    print(f"Negative TDOA sum: {negative_tau_sum:+.8e} s | abs={negative_abs:.8e}")
    print(f"Greater side by absolute sum: {dominant_side}")

    # Expose sum-comparison result to caller for downstream actions.
    best_result["positive_tau_sum"] = positive_tau_sum
    best_result["negative_tau_sum"] = negative_tau_sum
    best_result["dominant_side"] = dominant_side

    print("\n--- Best Combination Result ---")
    print(f"Best parameters: FFT_SIZE={best_result['fft_size']}, FFT_ADVANCE_STEP={best_result['advance_step']}")
    print(f"Highest score: {best_result['score']:.6f}")
    print(f"Final TDOA: {best_result['tau'] * 1e6:.2f} us")
    print(f"Distance difference: {best_result['distance_diff_cm']:.2f} cm")
    print(f"Direction: {best_result['direction']}")

    return best_result, results

def main():
    device_index = 1
    fs = 44100

    print("Starting continuous microphone monitoring...")
    print("All received sound is judged by overall TDOA voting.")
    print("Press Ctrl+C to stop.")

    wav_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "wav/13360.wav")

    try:
        while True:
            # Step 1: record exactly 2 seconds.
            audio_clean = record_audio_block(fs, RECORD_DURATION_SEC, device_index)
            print("\n[Processing] 2-second audio block")

            # Step 2: keep only blocks containing high-frequency content above 20kHz.
            has_hf, hf_ratio = has_high_frequency_content(audio_clean, fs)
            if not has_hf:
                print(f"Skip block: >20kHz energy ratio too low ({hf_ratio:.6f})")
                continue
            print(f"High-frequency content detected (>20kHz), energy ratio={hf_ratio:.6f}")

            ch1 = audio_clean[:, 0]
            ch2 = audio_clean[:, 1]

            # Step 3: run 5 rounds of overall TDOA voting.
            best_result, _ = run_parameter_sweep(ch1, ch2, fs)
            if best_result is None:
                continue

            # Step 4: play alert if positive TDOA absolute sum is greater than negative.
            if best_result.get("dominant_side") == "positive":
                play_alert_sound(wav_path)

    except KeyboardInterrupt:
        print("\nStopped by user.")

if __name__ == "__main__":
    main()