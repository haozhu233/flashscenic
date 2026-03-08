#!/usr/bin/env bash
#SBATCH --job-name=01_preprocess
#SBATCH --output=/cluster/scratch/gcardenal/flashscenic/experiments/batch_benchmark/logs/01_preprocess/%x_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --mem-per-cpu=64G
# No GPU — QC, normalization, HVG selection, PCA are all CPU-only.
# 8 × 32G = 256G RAM: needed for ad_neurons (424K cells, two dense layers + PCA).

set -euo pipefail

BENCH_DIR=/cluster/scratch/gcardenal/flashscenic/experiments/batch_benchmark
cd "$BENCH_DIR"
source /cluster/scratch/gcardenal/scanpy/bin/activate

echo "Node: $(hostname) | Date: $(date)"
python scripts/01_preprocess.py --dataset all
echo "Done: $(date)"
