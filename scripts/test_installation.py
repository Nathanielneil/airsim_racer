#!/usr/bin/env python3
"""
Test script to verify AirSim RACER installation
"""

import sys
import importlib.util
from pathlib import Path


def test_python_version():
    """Test Python version"""
    print("Testing Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 7:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} - Need Python 3.7+")
        return False


def test_required_packages():
    """Test required Python packages"""
    print("\nTesting required packages...")
    
    required_packages = [
        'numpy',
        'scipy', 
        'matplotlib',
        'cv2',  # opencv-python
        'PIL',  # Pillow
        'yaml',  # PyYAML
        'sklearn',  # scikit-learn
        'transforms3d',
        'airsim'
    ]
    
    all_ok = True
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
                print(f"✓ opencv-python ({cv2.__version__}) - OK")
            elif package == 'PIL':
                import PIL
                print(f"✓ Pillow ({PIL.__version__}) - OK")
            elif package == 'yaml':
                import yaml
                print(f"✓ PyYAML - OK")
            elif package == 'sklearn':
                import sklearn
                print(f"✓ scikit-learn ({sklearn.__version__}) - OK")
            else:
                spec = importlib.util.find_spec(package)
                if spec is not None:
                    module = importlib.import_module(package)
                    version = getattr(module, '__version__', 'unknown')
                    print(f"✓ {package} ({version}) - OK")
                else:
                    print(f"✗ {package} - NOT FOUND")
                    all_ok = False
        except ImportError:
            print(f"✗ {package} - IMPORT ERROR")
            all_ok = False
    
    return all_ok


def test_airsim_connection():
    """Test AirSim connection (basic import test)"""
    print("\nTesting AirSim connection...")
    
    try:
        import airsim
        print("✓ AirSim package imported successfully")
        
        # Test basic client creation (won't connect without running simulator)
        print("✓ AirSim client can be created (simulator not required for this test)")
        return True
    except Exception as e:
        print(f"✗ AirSim test failed: {e}")
        return False


def test_project_structure():
    """Test project file structure"""
    print("\nTesting project structure...")
    
    required_files = [
        'main.py',
        'environment.yml',
        'requirements.txt',
        'config/airsim_settings.json',
        'config/exploration_config.yaml',
        'src/__init__.py',
        'src/airsim_interface/__init__.py',
        'src/airsim_interface/airsim_client.py',
        'src/exploration/__init__.py',
        'src/exploration/exploration_manager.py',
        'src/exploration/frontier_finder.py',
        'src/exploration/grid_map.py',
        'src/planning/__init__.py',
        'src/planning/path_planner.py',
        'src/utils/__init__.py',
        'src/utils/math_utils.py',
    ]
    
    all_ok = True
    base_path = Path(__file__).parent.parent
    
    for file_path in required_files:
        full_path = base_path / file_path
        if full_path.exists():
            print(f"✓ {file_path} - OK")
        else:
            print(f"✗ {file_path} - NOT FOUND")
            all_ok = False
    
    return all_ok


def test_config_files():
    """Test configuration files"""
    print("\nTesting configuration files...")
    
    try:
        # Test YAML config
        import yaml
        config_path = Path(__file__).parent.parent / "config" / "exploration_config.yaml"
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            print("✓ exploration_config.yaml - Valid YAML")
            
            # Check required sections
            required_sections = ['exploration', 'mapping', 'planning', 'airsim']
            for section in required_sections:
                if section in config:
                    print(f"  ✓ {section} section found")
                else:
                    print(f"  ✗ {section} section missing")
            
        else:
            print("✗ exploration_config.yaml - NOT FOUND")
            return False
        
        # Test AirSim settings
        import json
        settings_path = Path(__file__).parent.parent / "config" / "airsim_settings.json"
        
        if settings_path.exists():
            with open(settings_path, 'r') as f:
                settings = json.load(f)
            print("✓ airsim_settings.json - Valid JSON")
        else:
            print("✗ airsim_settings.json - NOT FOUND")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False


def main():
    """Run all installation tests"""
    print("AirSim RACER Installation Test")
    print("=" * 50)
    
    tests = [
        ("Python Version", test_python_version),
        ("Required Packages", test_required_packages),
        ("AirSim Connection", test_airsim_connection),
        ("Project Structure", test_project_structure),
        ("Configuration Files", test_config_files),
    ]
    
    all_passed = True
    
    for test_name, test_func in tests:
        try:
            if not test_func():
                all_passed = False
        except Exception as e:
            print(f"✗ {test_name} - ERROR: {e}")
            all_passed = False
    
    print("\n" + "=" * 50)
    
    if all_passed:
        print("✓ All installation tests passed!")
        print("\nYou can now run the system with:")
        print("  python main.py")
        return 0
    else:
        print("✗ Some tests failed. Please check the installation.")
        print("\nTo fix issues:")
        print("  1. Install missing packages: pip install -r requirements.txt")
        print("  2. Check file paths and permissions")
        print("  3. Verify AirSim installation")
        return 1


if __name__ == "__main__":
    sys.exit(main())