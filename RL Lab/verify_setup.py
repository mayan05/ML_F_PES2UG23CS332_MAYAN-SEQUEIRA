#!/usr/bin/env python3
"""
Quick verification script to check if all dependencies are installed correctly.
Run this before starting the lab.
"""

import sys

def check_import(module_name, package_name=None):
    try:
        __import__(module_name)
        print(f"✓ {package_name or module_name} installed")
        return True
    except ImportError:
        print(f"✗ {package_name or module_name} NOT installed")
        return False

def main():
    print("Checking dependencies...")
    print("-" * 40)
    
    checks = [
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("cv2", "OpenCV"),
        ("matplotlib", "Matplotlib"),
        ("gymnasium", "Gymnasium"),
        ("gym_super_mario_bros", "gymnasium-super-mario-bros"),
        ("nes_py", "nes-py"),
    ]
    
    all_ok = True
    for module, name in checks:
        if not check_import(module, name):
            all_ok = False
    
    print("-" * 40)
    if all_ok:
        print("All dependencies installed correctly!")
        print("You can now proceed with the lab.")
        return 0
    else:
        print("Some dependencies are missing.")
        print("Please run: pip install -r requirements.txt")
        return 1

if __name__ == "__main__":
    sys.exit(main())

