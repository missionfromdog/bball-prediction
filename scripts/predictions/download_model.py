"""
Download best model if not available locally
Fallback for LFS issues
"""
import os
from pathlib import Path
import urllib.request
import sys

MODEL_PATH = Path(__file__).resolve().parents[2] / 'models' / 'histgradient_vegas_calibrated.pkl'

def is_lfs_pointer(filepath):
    """Check if file is a Git LFS pointer"""
    try:
        with open(filepath, 'r') as f:
            first_line = f.readline()
            return first_line.startswith('version https://git-lfs')
    except:
        return False

def download_from_github_release():
    """Download model from GitHub release"""
    # For now, just print message
    # In production, you'd have a GitHub release with the model
    print("Note: Model should be in repo. Check if LFS is properly disabled.")
    return False

def main():
    if not MODEL_PATH.exists():
        print(f"❌ Model not found: {MODEL_PATH}")
        sys.exit(1)
    
    if is_lfs_pointer(MODEL_PATH):
        print(f"⚠️  Model is LFS pointer, not actual file!")
        print(f"   File: {MODEL_PATH}")
        sys.exit(1)
    
    print(f"✅ Model file is valid (not LFS pointer)")
    print(f"   Size: {MODEL_PATH.stat().st_size / (1024*1024):.1f} MB")
    return True

if __name__ == '__main__':
    main()
