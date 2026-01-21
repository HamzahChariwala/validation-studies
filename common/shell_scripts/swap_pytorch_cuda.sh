#!/bin/bash
# swap_pytorch_cuda.sh
#
# Uninstall PyTorch and reinstall with a specific CUDA version.
#
# Usage:
#   ./swap_pytorch_cuda.sh 118   # CUDA 11.8
#   ./swap_pytorch_cuda.sh 121   # CUDA 12.1
#   ./swap_pytorch_cuda.sh 124   # CUDA 12.4

set -e

show_help() {
    echo "PyTorch CUDA Version Switcher"
    echo ""
    echo "Usage: $0 <cuda_version>"
    echo ""
    echo "Supported CUDA versions:"
    echo "  118 - CUDA 11.8 (Driver >= 450)"
    echo "  121 - CUDA 12.1 (Driver >= 525)"
    echo "  124 - CUDA 12.4 (Driver >= 550)"
}

if [ "$1" = "-h" ] || [ "$1" = "--help" ] || [ $# -eq 0 ]; then
    show_help
    exit 0
fi

CUDA_VERSION=$1

case $CUDA_VERSION in
    118)
        CUDA_DISPLAY="11.8"
        INDEX_URL="https://download.pytorch.org/whl/cu118"
        ;;
    121)
        CUDA_DISPLAY="12.1"
        INDEX_URL="https://download.pytorch.org/whl/cu121"
        ;;
    124)
        CUDA_DISPLAY="12.4"
        INDEX_URL="https://download.pytorch.org/whl/cu124"
        ;;
    *)
        echo "Error: Invalid CUDA version '${CUDA_VERSION}'"
        show_help
        exit 1
        ;;
esac

if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "Error: Python not found"
    exit 1
fi

PYTHON_CMD=$(command -v python3 || command -v python)

if ! $PYTHON_CMD -m pip --version &> /dev/null; then
    echo "Error: pip not found"
    exit 1
fi

echo "Switching to PyTorch with CUDA ${CUDA_DISPLAY}"

if $PYTHON_CMD -c "import torch" 2>/dev/null; then
    echo "Uninstalling current PyTorch..."
    $PYTHON_CMD -m pip uninstall -y torch torchvision torchaudio -q 2>/dev/null || true
fi

echo "Installing PyTorch with CUDA ${CUDA_DISPLAY}..."
$PYTHON_CMD -m pip install torch torchvision torchaudio --index-url "${INDEX_URL}" -q

echo ""
$PYTHON_CMD -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if hasattr(torch.version, 'cuda') and torch.version.cuda:
    print(f'CUDA version: {torch.version.cuda}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    # Quick test
    import time
    a = torch.randn(1024, 1024, device='cuda')
    b = torch.randn(1024, 1024, device='cuda')
    torch.cuda.synchronize()
    start = time.time()
    c = torch.matmul(a, b)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f'Test matmul (1024x1024): {elapsed*1000:.2f}ms')
"

echo ""
echo "Ready to run profiling."

