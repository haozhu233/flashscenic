#!/usr/bin/env bash
#SBATCH --job-name=00_validate
#SBATCH --output=/cluster/scratch/gcardenal/flashscenic/experiments/batch_benchmark/logs/00_validate/%x_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=32G

set -euo pipefail

BENCH_DIR=/cluster/scratch/gcardenal/flashscenic/experiments/batch_benchmark
cd "$BENCH_DIR"
source /cluster/scratch/gcardenal/scanpy/bin/activate

echo "Node: $(hostname) | Date: $(date)"
python scripts/00_validate_data.py --dataset all
echo "Done: $(date)"
