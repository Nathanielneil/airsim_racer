#!/usr/bin/env python3
"""
Script to help install and configure AirSim for Windows
"""

import os
import sys
import urllib.request
import zipfile
import shutil
from pathlib import Path


def download_file(url: str, filename: str) -> bool:
    """Download a file from URL"""
    try:
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
        print(f"Downloaded {filename}")
        return True
    except Exception as e:
        print(f"Error downloading {filename}: {e}")
        return False


def extract_zip(zip_path: str, extract_to: str) -> bool:
    """Extract zip file"""
    try:
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Extracted to {extract_to}")
        return True
    except Exception as e:
        print(f"Error extracting {zip_path}: {e}")
        return False


def setup_airsim_config():
    """Setup AirSim configuration directory"""
    try:
        # Get user documents directory
        documents_dir = Path.home() / "Documents"
        airsim_dir = documents_dir / "AirSim"
        
        # Create AirSim directory if it doesn't exist
        airsim_dir.mkdir(exist_ok=True)
        
        # Copy settings file
        config_source = Path("config/airsim_settings.json")
        config_dest = airsim_dir / "settings.json"
        
        if config_source.exists():
            shutil.copy2(config_source, config_dest)
            print(f"Copied AirSim settings to {config_dest}")
        else:
            print("Warning: AirSim settings file not found")
        
        return True
        
    except Exception as e:
        print(f"Error setting up AirSim config: {e}")
        return False


def main():
    print("AirSim Installation Helper")
    print("=" * 50)
    
    print("\nNote: This script helps with AirSim configuration.")
    print("You still need to download AirSim binaries manually from:")
    print("https://github.com/Microsoft/AirSim/releases")
    print()
    
    # Setup configuration
    if setup_airsim_config():
        print("✓ AirSim configuration setup complete")
    else:
        print("✗ AirSim configuration setup failed")
        return 1
    
    print("\nNext steps:")
    print("1. Download AirSim binary from GitHub releases")
    print("2. Extract to a folder (e.g., C:\\AirSim)")
    print("3. Run the .exe file to start AirSim")
    print("4. Use 'conda activate airsim_racer' and 'python main.py' to start exploration")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())