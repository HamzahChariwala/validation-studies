#!/bin/bash
# setup.sh
#
# One-step setup for validation-studies environment
# Creates venv and installs all dependencies

set -e

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found"
    exit 1
fi

# Check if python3-venv package is installed (Debian/Ubuntu)
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)

if command -v dpkg &> /dev/null; then
    if ! dpkg -l python${PYTHON_VERSION}-venv 2>/dev/null | grep -q '^ii'; then
        echo "python${PYTHON_VERSION}-venv package not found. Installing..."
        sudo apt update -qq
        sudo apt install -y python${PYTHON_VERSION}-venv
        echo "Successfully installed python${PYTHON_VERSION}-venv"
    fi
fi

echo "Creating virtual environment..."
python3 -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

echo "Upgrading pip..."
python -m pip install --upgrade pip -q

echo "Installing base dependencies..."
pip install numpy>=2.0.0 tensorboard>=2.20.0 scipy>=1.7.0 scikit-learn>=1.0.0 PyYAML>=6.0.0 pandas>=2.0.0 matplotlib>=3.7.0 seaborn>=0.12.0 -q

echo ""
echo "Select CUDA version:"
echo "  1) CUDA 11.8 (Driver >= 450)"
echo "  2) CUDA 12.1 (Driver >= 525)"
echo "  3) CUDA 12.4 (Driver >= 550)"
echo ""
read -p "Choose [1-3]: " choice

case $choice in
    1)
        echo "Installing PyTorch with CUDA 11.8..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -q
        ;;
    2)
        echo "Installing PyTorch with CUDA 12.1..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q
        ;;
    3)
        echo "Installing PyTorch with CUDA 12.4..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 -q
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "Setup complete."
echo ""

# Verify installation
if python -c "import torch" 2>/dev/null; then
    python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if hasattr(torch.version, 'cuda') and torch.version.cuda:
    print(f'CUDA version: {torch.version.cuda}')
"
fi

echo ""
echo "To activate in new terminals: source venv/bin/activate"
echo "To run profiling: cd compute && python profile_matmul.py"
echo "To switch CUDA versions: ./swap_pytorch_cuda.sh <118|121|124>"

