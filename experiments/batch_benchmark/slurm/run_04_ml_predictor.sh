#!/usr/bin/env bash
#SBATCH --job-name=04_ml_predictor
#SBATCH --output=/cluster/scratch/gcardenal/flashscenic/experiments/batch_benchmark/logs/04_ml_predictor/%x_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=16G
# No GPU — ElasticNet (sklearn) is CPU-only.
# 16 × 16G = 256G: loading embeddings for 424K cells across 4 methods.
# Run AFTER all three 02x jobs have completed.

set -euo pipefail

BENCH_DIR=/cluster/scratch/gcardenal/flashscenic/experiments/batch_benchmark
cd "$BENCH_DIR"
source /cluster/scratch/gcardenal/scanpy/bin/activate

DATASET="${DATASET:-all}"

echo "Node: $(hostname) | Date: $(date)"
export OMP_NUM_THREADS=4
python -u scripts/04_ml_predictor.py --dataset "$DATASET" --tasks sex celltype age
echo "Done: $(date)"