@echo off
REM Run single drone exploration

if "%1"=="" (
    echo Usage: run_single_drone.bat [drone_id]
    echo Example: run_single_drone.bat 0
    pause
    exit /b 1
)

call conda activate airsim_racer
python main.py --drone-id %1

pause