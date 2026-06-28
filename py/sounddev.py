import sounddevice as sd

def list_microphones():
    print("Available audio input devices:\n")
    devices = sd.query_devices()
    for idx, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            print(f"Index: {idx}")
            print(f"  Name: {device['name']}")
            print(f"  Host API: {sd.query_hostapis(device['hostapi'])['name']}")
            print(f"  Max input channels: {device['max_input_channels']}")
            print(f"  Default samplerate: {device['default_samplerate']}")
            print("-" * 40)

if __name__ == "__main__":
    list_microphones()