#!/bin/bash
# Convenience script to activate the virtual environment

echo "🏀 Activating NBA Prediction Project virtual environment..."
source venv/bin/activate

echo "✅ Virtual environment activated!"
echo ""
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""
echo "Quick commands:"
echo "  jupyter notebook           - Open Jupyter notebooks"
echo "  streamlit run src/streamlit_app.py - Run the web app"
echo "  python test_installation.py - Test installation"
echo "  deactivate                 - Exit virtual environment"
echo ""

