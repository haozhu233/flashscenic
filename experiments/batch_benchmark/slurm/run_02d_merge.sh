#!/usr/bin/env bash
#SBATCH --job-name=02d_merge
#SBATCH --output=/cluster/scratch/gcardenal/flashscenic/experiments/batch_benchmark/logs/02d_merge/%x_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=20G
# Merges all embedding .npy files into the preprocessed h5ad.
# Runs after 02a, 02b, 02c complete — no GPU needed.

set -euo pipefail

BENCH_DIR=/cluster/scratch/gcardenal/flashscenic/experiments/batch_benchmark
cd "$BENCH_DIR"
source /cluster/scratch/gcardenal/scanpy/bin/activate

mkdir -p "$BENCH_DIR/logs/02d_merge"

echo "Node: $(hostname) | Date: $(date)"
python scripts/02d_merge_embeddings.py --dataset all
echo "Done: $(date)"
