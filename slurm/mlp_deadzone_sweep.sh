#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --exclude=hendrixgpu20fl
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --job-name=objection-classifier
#SBATCH --output=slurm-%j.log


set -euo pipefail

export PYTHONUNBUFFERED=1

# node/GPU layout
hostname
nvidia-smi

# Load modules & conda
module load miniconda
module load cuda

conda_environment="codeq"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$conda_environment"

which python
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cudnn', torch.backends.cudnn.version())"

echo "=== Launching 8 MLP dead-zone sweep runs in parallel ==="

CONFIGS=(
    configs/baseline_mlp.yaml
    configs/deadzone_mlp.yaml
    configs/deadzone_mlp_group.yaml
)

PIDS=()
for cfg in "${CONFIGS[@]}"; do
    echo "Starting: $cfg"
    python run_training.py --config "$cfg" --device cuda &
    PIDS+=($!)
done

echo "All ${#PIDS[@]} runs launched. PIDs: ${PIDS[*]}"

# Wait for all; track failures
FAILED=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        echo "FAILED: PID $pid"
        FAILED=$((FAILED + 1))
    fi
done

if [ $FAILED -ne 0 ]; then
    echo "$FAILED / ${#PIDS[@]} runs failed."
    exit 1
fi

echo "=== All ${#PIDS[@]} runs completed successfully ==="
