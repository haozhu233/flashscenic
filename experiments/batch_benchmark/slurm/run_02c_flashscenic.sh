#!/usr/bin/env bash
#SBATCH --job-name=02c_flashscenic
#SBATCH --output=/cluster/scratch/gcardenal/flashscenic/experiments/batch_benchmark/logs/02c_flashscenic/%x_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=a100_80gb:2
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=20G
# GRN inference: subsampled cells for GRN learning (RTX 4090 24GB limit).
# Regulons learned on subsampled cells are then scored on the full dataset via AUCell.

set -euo pipefail

BENCH_DIR=/cluster/scratch/gcardenal/flashscenic/experiments/batch_benchmark
cd "$BENCH_DIR"
source /cluster/scratch/gcardenal/scanpy/bin/activate

CACHE_DIR="$BENCH_DIR/flashscenic_data"

# Abort early if ranking DBs are missing — avoids silent network failures
if [ -z "$(ls "$CACHE_DIR"/*.feather 2>/dev/null)" ]; then
    echo "ERROR: No .feather ranking databases found in $CACHE_DIR"
    echo "Run on the login node first: bash slurm/00_prefetch_resources.sh"
    exit 1
fi

echo "Node: $(hostname) | Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'none')"
echo "Cache dir: $CACHE_DIR ($(ls "$CACHE_DIR" | wc -l) files)"

#echo "--- ad_inhibitory ---"
#python scripts/02c_run_flashscenic.py --dataset ad_inhibitory --cache_dir "$CACHE_DIR"

echo "--- als_motor_cortex ---"
python scripts/02c_run_flashscenic.py --dataset als_motor_cortex --cache_dir "$CACHE_DIR"

echo "--- als_spinal_cord ---"
python scripts/02c_run_flashscenic.py --dataset als_spinal_cord --cache_dir "$CACHE_DIR"

echo "Done: $(date)"
