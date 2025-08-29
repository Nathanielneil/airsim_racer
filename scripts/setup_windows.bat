@echo off
REM AirSim RACER Windows Setup Script

echo Setting up AirSim RACER environment...

REM Check if conda is installed
conda --version >nul 2>&1
if errorlevel 1 (
    echo Error: Conda is not installed or not in PATH
    echo Please install Anaconda or Miniconda first
    pause
    exit /b 1
)

REM Create conda environment
echo Creating conda environment...
conda env create -f environment.yml

REM Activate environment
echo Activating environment...
call conda activate airsim_racer

REM Install additional dependencies
echo Installing additional dependencies...
pip install -r requirements.txt

REM Copy AirSim settings to user directory
echo Setting up AirSim configuration...
set AIRSIM_CONFIG_DIR=%USERPROFILE%\Documents\AirSim
if not exist "%AIRSIM_CONFIG_DIR%" mkdir "%AIRSIM_CONFIG_DIR%"
copy "config\airsim_settings.json" "%AIRSIM_CONFIG_DIR%\settings.json"

echo.
echo Setup completed successfully!
echo.
echo To run the system:
echo 1. Start AirSim simulation (UE4 executable)
echo 2. Run: conda activate airsim_racer
echo 3. Run: python main.py
echo.
pause