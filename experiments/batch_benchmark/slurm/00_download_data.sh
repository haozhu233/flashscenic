#!/usr/bin/env bash
# Download scIB benchmark datasets and CellxGENE AD dataset.
#
# Sources:
#   figshare article 12420968:
#     25717328  Immune_ALL_human.h5ad                2064 MB
#     24539828  human_pancreas_norm_complexBatch.h5ad  316 MB
#   CellxGENE collection 0d35c0fd (AD resilience, inhibitory neurons):
#     e7be14ca-...  ad_inhibitory.h5ad                1070 MB  (has sex: M/F)

set -euo pipefail

DATA_DIR="$(dirname "$0")/../data"
mkdir -p "$DATA_DIR"

# Download from figshare via the API endpoint (avoids 202 async queueing)
download_figshare() {
    local file_id="$1"
    local dest="$2"
    local min_mb="$3"

    if [ -f "$dest" ]; then
        local size_mb
        size_mb=$(du -m "$dest" | cut -f1)
        if [ "$size_mb" -ge "$min_mb" ]; then
            echo "[skip] $(basename "$dest") already exists (${size_mb} MB)"
            return
        else
            echo "Existing $(basename "$dest") is too small (${size_mb} MB) — re-downloading"
            rm -f "$dest"
        fi
    fi

    local url="https://api.figshare.com/v2/file/download/${file_id}"
    echo "Downloading $(basename "$dest") from figshare (ID ${file_id}) ..."
    wget --progress=bar:force:noscroll --no-check-certificate \
         --tries=3 --waitretry=10 -L -O "$dest" "$url"

    local size_mb
    size_mb=$(du -m "$dest" | cut -f1)
    if [ "$size_mb" -lt "$min_mb" ]; then
        echo "ERROR: $(basename "$dest") is only ${size_mb} MB — download failed."
        rm -f "$dest"; exit 1
    fi
    echo "Saved: $dest (${size_mb} MB)"
}

# Download directly from a URL (CellxGENE uses direct S3 links)
download_direct() {
    local url="$1"
    local dest="$2"
    local min_mb="$3"

    if [ -f "$dest" ]; then
        local size_mb
        size_mb=$(du -m "$dest" | cut -f1)
        if [ "$size_mb" -ge "$min_mb" ]; then
            echo "[skip] $(basename "$dest") already exists (${size_mb} MB)"
            return
        else
            echo "Existing $(basename "$dest") is too small (${size_mb} MB) — re-downloading"
            rm -f "$dest"
        fi
    fi

    echo "Downloading $(basename "$dest") ..."
    wget --progress=bar:force:noscroll --no-check-certificate \
         --tries=3 --waitretry=10 -L -O "$dest" "$url"

    local size_mb
    size_mb=$(du -m "$dest" | cut -f1)
    if [ "$size_mb" -lt "$min_mb" ]; then
        echo "ERROR: $(basename "$dest") is only ${size_mb} MB — download failed."
        rm -f "$dest"; exit 1
    fi
    echo "Saved: $dest (${size_mb} MB)"
}

echo "=== Downloading Immune_ALL_human.h5ad (scIB) ==="
download_figshare "25717328" "$DATA_DIR/immune_human.h5ad" 500

echo ""
echo "=== Downloading human_pancreas_norm_complexBatch.h5ad (scIB) ==="
download_figshare "24539828" "$DATA_DIR/pancreas.h5ad" 100

echo ""
echo "=== Downloading AD neurons — all cells (CellxGENE, 424K cells, ~6.4 GB, has sex) ==="
download_direct \
    "https://datasets.cellxgene.cziscience.com/9066d7f5-924e-4022-9c8a-ceaff3d50104.h5ad" \
    "$DATA_DIR/ad_neurons.h5ad" \
    500

echo ""
echo "=== Downloading AD inhibitory neurons (CellxGENE, ~1070 MB, has sex) ==="
download_direct \
    "https://datasets.cellxgene.cziscience.com/e7be14ca-e499-4dfa-8292-0768896852dd.h5ad" \
    "$DATA_DIR/ad_inhibitory.h5ad" \
    500

echo ""
echo "Downloads complete. Now run:"
echo "  python scripts/00_validate_data.py --dataset all"
