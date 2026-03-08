#!/usr/bin/env bash
#SBATCH --job-name=02a_baselines
#SBATCH --output=/cluster/scratch/gcardenal/flashscenic/experiments/batch_benchmark/logs/02a_baselines/%x_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=16G
# No GPU — raw PCA copy and Harmony are CPU-only.
# 16 × 16G = 256G RAM for loading all three preprocessed datasets.

set -euo pipefail

BENCH_DIR=/cluster/scratch/gcardenal/flashscenic/experiments/batch_benchmark
cd "$BENCH_DIR"
source /cluster/scratch/gcardenal/scanpy/bin/activate

echo "Node: $(hostname) | Date: $(date)"
python scripts/02a_run_baselines.py --dataset all
echo "Done: $(date)"
