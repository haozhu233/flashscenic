#!/usr/bin/env bash
#SBATCH --job-name=03_metrics
#SBATCH --output=/cluster/scratch/gcardenal/flashscenic/experiments/batch_benchmark/logs/03_metrics/%x_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=16G
# No GPU — LISI, ASW, and Leiden clustering are CPU-only.
# 16 × 16G = 256G: k-NN graph on 424K cells is memory-heavy.
# Run AFTER all three 02x jobs have completed.

set -euo pipefail

BENCH_DIR=/cluster/scratch/gcardenal/flashscenic/experiments/batch_benchmark
cd "$BENCH_DIR"
source /cluster/scratch/gcardenal/scanpy/bin/activate

echo "Node: $(hostname) | Date: $(date)"
python scripts/03_compute_metrics.py --dataset all
echo "Done: $(date)"
