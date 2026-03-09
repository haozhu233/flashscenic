#!/usr/bin/env bash
#SBATCH --job-name=05_visualize
#SBATCH --output=/cluster/scratch/gcardenal/flashscenic/experiments/batch_benchmark/logs/05_visualize/%x_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --mem-per-cpu=24G
# No GPU — UMAP is CPU-only (4 methods × 3 datasets = 12 UMAP runs).
# 8 × 24G = 192G: UMAP on 424K cells needs significant RAM.
# Run AFTER steps 03 and 04 have completed.

set -euo pipefail

BENCH_DIR=/cluster/scratch/gcardenal/flashscenic/experiments/batch_benchmark
cd "$BENCH_DIR"
source /cluster/scratch/gcardenal/scanpy/bin/activate

DATASET="${DATASET:-all}"

echo "Node: $(hostname) | Date: $(date)"
export MPLBACKEND=Agg
python -u scripts/05_visualize.py --dataset "$DATASET"
echo "Done: $(date)"
