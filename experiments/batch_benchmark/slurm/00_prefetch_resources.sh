#!/usr/bin/env bash
# Pre-download flashSCENIC resources on the login node (which has internet).
# Run this ONCE before submitting run_02c_flashscenic.sh.
# Compute nodes have no internet access and will fail if resources are missing.
#
# Usage (on login node):
#   source /cluster/scratch/gcardenal/scanpy/bin/activate
#   bash 00_prefetch_resources.sh

set -euo pipefail

BENCH_DIR=/cluster/scratch/gcardenal/flashscenic/experiments/batch_benchmark
CACHE_DIR="$BENCH_DIR/flashscenic_data"
mkdir -p "$CACHE_DIR"

cd "$BENCH_DIR"
source /cluster/scratch/gcardenal/scanpy/bin/activate

echo "Pre-fetching flashSCENIC resources to: $CACHE_DIR"
echo "This downloads ~2-4 GB of ranking databases. This may take several minutes."
echo ""

python - <<'EOF'
import sys
sys.path.insert(0, ".")
from flashscenic.data import download_data

for species in ["human"]:
    print(f"Downloading resources for: {species}")
    resources = download_data(
        species=species,
        version="v10",
        datasource="scenic",
        cache_dir="flashscenic_data",
    )
    print(f"  TF list:          {resources.tf_list}")
    print(f"  Ranking DBs:      {resources.ranking_dbs}")
    print(f"  Motif annotation: {resources.motif_annotation}")
    print()

print("All resources cached. Ready to submit run_02c_flashscenic.sh")
EOF