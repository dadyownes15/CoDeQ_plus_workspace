#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --exclude=hendrixgpu20fl,hendrixgpu26fl
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --job-name=mlp-dz-sweep
#SBATCH --output=slurm-%A_%a.log
#SBATCH --array=0-3

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

CONFIGS=(
    configs/baseline_mlp.yaml
    configs/deadzone_mlp.yaml
    configs/deadzone_mlp_group.yaml
    configs/deadzone_mlp_group_soft.yaml
)

echo "=== Array task ${SLURM_ARRAY_TASK_ID}: ${CONFIGS[$SLURM_ARRAY_TASK_ID]} ==="
python run_training.py --config "${CONFIGS[$SLURM_ARRAY_TASK_ID]}" --device cuda
echo "=== Done ==="
