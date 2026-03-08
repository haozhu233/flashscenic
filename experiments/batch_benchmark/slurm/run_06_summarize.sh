#!/usr/bin/env bash
#SBATCH --job-name=06_summarize
#SBATCH --output=/cluster/scratch/gcardenal/flashscenic/experiments/batch_benchmark/logs/06_summarize/%x_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=8G
# Lightweight — reads CSVs and writes SUMMARY.md.
# Run AFTER steps 03, 04, and 05 have completed.

set -euo pipefail

BENCH_DIR=/cluster/scratch/gcardenal/flashscenic/experiments/batch_benchmark
cd "$BENCH_DIR"
source /cluster/scratch/gcardenal/scanpy/bin/activate

echo "Node: $(hostname) | Date: $(date)"
python scripts/06_summarize.py --dataset all
echo "Done: $(date)"
echo "Results: $BENCH_DIR/results/SUMMARY.md"
