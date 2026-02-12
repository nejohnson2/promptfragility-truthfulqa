#!/bin/bash
#SBATCH --job-name=pf-test
#SBATCH --output=pf-test_%j.out
#SBATCH --error=pf-test_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=00:10:00
#SBATCH --partition=debug-h200x4
#
# Quick test script — validates environment, conda, GPU, and Python packages.
# Does NOT run any model inference.
#
# Usage:
#   sbatch scripts/slurm_test.sbatch

echo "=== Basic Info ==="
echo "Job ID:    ${SLURM_JOB_ID:-local}"
echo "Node:      $(hostname)"
echo "Date:      $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "User:      $USER"
echo "Shell:     $SHELL"
echo "PWD:       $(pwd)"
echo ""

echo "=== Step 1: Module load ==="
module load cuda12.8/toolkit/12.8.0 && echo "OK: cuda module loaded" || echo "FAIL: cuda module"
echo ""

echo "=== Step 2: Conda init ==="
eval "$(conda shell.bash hook)" && echo "OK: conda hook" || echo "FAIL: conda hook"
echo ""

echo "=== Step 3: Conda activate ==="
conda activate promptfragility && echo "OK: conda activate" || echo "FAIL: conda activate"
echo "Conda env: ${CONDA_DEFAULT_ENV:-NOT SET}"
echo "Python:    $(which python 2>&1)"
echo ""

echo "=== Step 4: Python version ==="
python --version 2>&1 || echo "FAIL: python not found"
echo ""

echo "=== Step 5: GPU ==="
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    echo "OK: nvidia-smi works"
else
    echo "FAIL: nvidia-smi not found"
fi
echo ""

echo "=== Step 6: Python packages ==="
python -c "
packages = [
    'torch', 'transformers', 'accelerate', 'datasets',
    'numpy', 'pandas', 'scipy', 'matplotlib', 'jinja2',
    'yaml', 'tqdm',
]
for pkg in packages:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, '__version__', 'ok')
        print(f'  OK: {pkg} ({ver})')
    except ImportError as e:
        print(f'  FAIL: {pkg} ({e})')
" || echo "FAIL: python -c crashed"
echo ""

echo "=== Step 7: CUDA in PyTorch ==="
python -c "
import torch
print(f'  torch.cuda.is_available(): {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version: {torch.version.cuda}')
    print(f'  GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)')
else:
    print('  WARNING: CUDA not available to PyTorch')
" || echo "FAIL: torch cuda check crashed"
echo ""

echo "=== Step 8: HuggingFace auth ==="
python -c "
from huggingface_hub import HfApi
api = HfApi()
try:
    info = api.whoami()
    print(f'  OK: logged in as {info[\"name\"]}')
except Exception as e:
    print(f'  FAIL: not authenticated ({e})')
" || echo "FAIL: huggingface check crashed"
echo ""

echo "=== Step 9: Project files ==="
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
for f in "data/truthfulqa_cache.jsonl" "prompts/conditions.yaml" "prompts/templates/truthfulqa_mc.jinja" "src/config.py" "src/eval/run_eval.py"; do
    if [ -f "$PROJECT_ROOT/$f" ]; then
        echo "  OK: $f"
    else
        echo "  MISSING: $f"
    fi
done
echo ""

echo "=== All checks complete ==="
