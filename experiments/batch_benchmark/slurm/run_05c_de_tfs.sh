#!/usr/bin/env bash
#SBATCH --job-name=05c_de_tfs
#SBATCH --output=/cluster/scratch/gcardenal/flashscenic/experiments/batch_benchmark/logs/05c_de_tfs/%x_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=16G
# No GPU — donor-level pseudobulk + Mann-Whitney U, CPU-only.

set -euo pipefail

BENCH_DIR=/cluster/scratch/gcardenal/flashscenic/experiments/batch_benchmark
cd "$BENCH_DIR"
source /cluster/scratch/gcardenal/scanpy/bin/activate

DATASET="${DATASET:-als_motor_cortex}"

echo "Node: $(hostname) | Date: $(date)"
python -u scripts/06_de_tfs.py --dataset "$DATASET"
echo "Done: $(date)"
