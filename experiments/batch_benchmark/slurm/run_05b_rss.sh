#!/usr/bin/env bash
#SBATCH --job-name=05b_rss
#SBATCH --output=/cluster/scratch/gcardenal/flashscenic/experiments/batch_benchmark/logs/05b_rss/%x_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=16G
# CPU-only: GMM binarization + RSS on up to 424K cells × 261 regulons.
# 8 × 16G = 128G RAM; runs after 05_visualize.

set -euo pipefail

BENCH_DIR=/cluster/scratch/gcardenal/flashscenic/experiments/batch_benchmark
cd "$BENCH_DIR"
source /cluster/scratch/gcardenal/scanpy/bin/activate

mkdir -p "$BENCH_DIR/logs/05b_rss"

echo "Node: $(hostname) | Date: $(date)"
python scripts/05b_rss.py --dataset all
echo "Done: $(date)"
