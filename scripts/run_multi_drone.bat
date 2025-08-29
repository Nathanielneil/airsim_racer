@echo off
REM Run multi-drone exploration

if "%1"=="" (
    echo Usage: run_multi_drone.bat [num_drones]
    echo Example: run_multi_drone.bat 3
    pause
    exit /b 1
)

call conda activate airsim_racer
python main.py --num-drones %1

pause