#!/bin/bash

################################################################################
# Launch script for point-to-point communication profiling
#
# Usage:
#   bash launch_p2p.sh [OPTIONS]
#
# Options:
#   --gpus N          Number of GPUs to use (default: all available)
#   --lock-clocks     Lock GPU clocks to max frequency (requires sudo)
#   --help            Show this help message
################################################################################

set -e  # Exit on error

# Default values
NUM_GPUS=""
LOCK_CLOCKS=false
OUTPUT_DIR="./traces"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --lock-clocks)
            LOCK_CLOCKS=true
            shift
            ;;
        --help)
            head -n 13 "$0" | tail -n 10
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Detect number of GPUs if not specified
if [ -z "$NUM_GPUS" ]; then
    NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
    echo "Auto-detected $NUM_GPUS GPUs"
fi

# Validate environment
echo "Validating environment..."
python -c "
import sys
sys.path.append('../../..')
from networking.utils import validate_environment
is_valid, issues = validate_environment()
if not is_valid:
    print('Environment validation failed:')
    for issue in issues:
        print(f'  - {issue}')
    sys.exit(1)
print('✓ Environment valid')
"

if [ $? -ne 0 ]; then
    echo "Environment validation failed. Please fix issues and try again."
    exit 1
fi

# Lock GPU clocks if requested
if [ "$LOCK_CLOCKS" = true ]; then
    echo ""
    echo "Locking GPU clocks (requires sudo)..."
    
    # Check if we have sudo
    if ! sudo -n true 2>/dev/null; then
        echo "This requires sudo access. You may be prompted for your password."
    fi
    
    # Lock clocks for each GPU
    for ((i=0; i<$NUM_GPUS; i++)); do
        echo "  Locking GPU $i..."
        
        # Enable persistence mode
        sudo nvidia-smi -i $i -pm 1 2>/dev/null || echo "    (persistence mode already enabled)"
        
        # Get max clock
        MAX_CLOCK=$(nvidia-smi -i $i --query-gpu=clocks.max.graphics --format=csv,noheader,nounits)
        
        # Lock to max
        sudo nvidia-smi -i $i -lgc $MAX_CLOCK 2>/dev/null || echo "    (could not lock clock)"
        
        echo "    ✓ GPU $i locked to $MAX_CLOCK MHz"
    done
    
    echo "GPU clocks locked"
fi

# Print configuration
echo ""
echo "========================================================================"
echo "POINT-TO-POINT COMMUNICATION PROFILING"
echo "========================================================================"
echo "GPUs: $NUM_GPUS"
echo "Output directory: $OUTPUT_DIR"
echo "Locked clocks: $LOCK_CLOCKS"
echo ""

# Show GPU status
echo "GPU Status:"
nvidia-smi --query-gpu=index,name,temperature.gpu,clocks.current.graphics,power.draw --format=csv,noheader | \
    awk -F', ' '{printf "  GPU %s: %s | %s°C | %s MHz | %s\n", $1, $2, $3, $4, $5}'
echo ""

# Check for running processes
RUNNING_PROCS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
if [ "$RUNNING_PROCS" -gt 0 ]; then
    echo "⚠ Warning: $RUNNING_PROCS GPU process(es) already running"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
fi

# Set environment variables for NCCL
echo "Setting NCCL environment..."
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,COLL

# Launch with torchrun
echo "========================================================================"
echo "Launching with torchrun..."
echo "========================================================================"
echo ""

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=1 \
    --node_rank=0 \
    profile_p2p.py

RETVAL=$?

echo ""
echo "========================================================================"

if [ $RETVAL -eq 0 ]; then
    echo "✓ PROFILING COMPLETED SUCCESSFULLY"
    echo "========================================================================"
    echo ""
    echo "Output files:"
    echo "  Directory: $OUTPUT_DIR"
    echo ""
    
    # List generated files
    if [ -d "$OUTPUT_DIR" ]; then
        echo "  Files:"
        ls -lh "$OUTPUT_DIR" | tail -n +2 | awk '{printf "    %s  %s\n", $9, $5}'
    fi
else
    echo "✗ PROFILING FAILED (exit code: $RETVAL)"
    echo "========================================================================"
fi

# Unlock GPU clocks if we locked them
if [ "$LOCK_CLOCKS" = true ]; then
    echo ""
    echo "Unlocking GPU clocks..."
    
    for ((i=0; i<$NUM_GPUS; i++)); do
        sudo nvidia-smi -i $i -rgc 2>/dev/null || echo "  Could not unlock GPU $i"
    done
    
    echo "GPU clocks unlocked"
fi

echo ""

exit $RETVAL

