@echo off
:start
echo [%date% %time%] 脚本已启动 >> log.txt
python py/ceilingsnd_phat.py
echo [%date% %time%] 脚本意外退出，将在3秒后重启... >> log.txt
timeout /t 3 /nobreak > NUL
goto start