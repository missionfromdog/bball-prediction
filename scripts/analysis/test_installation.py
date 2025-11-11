"""
Test script to verify all key dependencies are properly installed
"""

import sys

def test_imports():
    """Test importing all critical packages"""
    tests_passed = []
    tests_failed = []
    
    # Core packages
    packages_to_test = [
        ('pandas', 'import pandas as pd'),
        ('numpy', 'import numpy as np'),
        ('scikit-learn', 'from sklearn.metrics import accuracy_score'),
        ('xgboost', 'import xgboost as xgb'),
        ('lightgbm', 'import lightgbm as lgb'),
        ('matplotlib', 'import matplotlib.pyplot as plt'),
        ('seaborn', 'import seaborn as sns'),
        ('streamlit', 'import streamlit as st'),
        ('optuna', 'import optuna'),
        ('shap', 'import shap'),
        ('selenium', 'from selenium import webdriver'),
        ('beautifulsoup4', 'from bs4 import BeautifulSoup'),
        ('joblib', 'import joblib'),
        ('neptune', 'import neptune'),
        ('plotly', 'import plotly.express as px'),
    ]
    
    print("Testing package imports...\n")
    print("=" * 60)
    
    for package_name, import_statement in packages_to_test:
        try:
            exec(import_statement)
            tests_passed.append(package_name)
            print(f"✅ {package_name:20s} - OK")
        except ImportError as e:
            tests_failed.append((package_name, str(e)))
            print(f"❌ {package_name:20s} - FAILED: {e}")
    
    print("=" * 60)
    print(f"\nResults: {len(tests_passed)}/{len(packages_to_test)} packages imported successfully")
    
    if tests_failed:
        print(f"\n⚠️  {len(tests_failed)} package(s) failed:")
        for pkg, error in tests_failed:
            print(f"   - {pkg}: {error}")
        return False
    else:
        print("\n🎉 All packages imported successfully!")
        return True

def test_src_modules():
    """Test importing local src modules"""
    print("\n" + "=" * 60)
    print("Testing local src modules...\n")
    
    modules_to_test = [
        'src.constants',
        'src.feature_engineering',
        'src.model_training',
        'src.data_processing',
    ]
    
    tests_passed = []
    tests_failed = []
    
    for module_name in modules_to_test:
        try:
            __import__(module_name)
            tests_passed.append(module_name)
            print(f"✅ {module_name:30s} - OK")
        except Exception as e:
            tests_failed.append((module_name, str(e)))
            print(f"❌ {module_name:30s} - FAILED: {e}")
    
    print("=" * 60)
    print(f"\nResults: {len(tests_passed)}/{len(modules_to_test)} modules imported successfully")
    
    if tests_failed:
        print(f"\n⚠️  {len(tests_failed)} module(s) failed:")
        for module, error in tests_failed:
            print(f"   - {module}: {error}")
        return False
    else:
        print("\n🎉 All src modules imported successfully!")
        return True

def check_python_version():
    """Check Python version"""
    print("\nPython Version Check:")
    print("=" * 60)
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 9:
        print("✅ Python version is compatible")
        return True
    else:
        print("⚠️  Python version should be 3.9 or higher")
        return False

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("NBA PREDICTION PROJECT - INSTALLATION TEST")
    print("=" * 60)
    
    # Test Python version
    py_ok = check_python_version()
    
    # Test package imports
    packages_ok = test_imports()
    
    # Test src modules
    modules_ok = test_src_modules()
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    
    all_ok = py_ok and packages_ok and modules_ok
    
    if all_ok:
        print("✅ All tests passed! Your environment is ready to use.")
        print("\nNext steps:")
        print("  1. Review the SETUP_GUIDE.md for usage instructions")
        print("  2. Open Jupyter notebooks to explore: jupyter notebook")
        print("  3. Run Streamlit app: streamlit run src/streamlit_app.py")
        sys.exit(0)
    else:
        print("⚠️  Some tests failed. Please review the errors above.")
        sys.exit(1)

