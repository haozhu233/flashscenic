#!/usr/bin/env bash
# =============================================================================
# run_experiments.sh — Robustness experiment pipeline for flashscenic
#
# Runs preprocessing once (skips if outputs already exist), then runs the
# three stability experiments in sequence.
#
# Usage:
#   sbatch run_experiments.sh [--n_runs N] [--n_steps S] [--device cuda|cpu]
#   bash   run_experiments.sh [--n_runs N] [--n_steps S] [--device cuda|cpu]
#
# All other parameters (species, paths, thresholds) are read from config.py.
# =============================================================================
#SBATCH --job-name=FlashScenic
#SBATCH --output=/orcd/data/omarabu/001/gonzalo/flashscenic_2/flashscenic/experiments/logs/%x_%j.out
#SBATCH --error=/orcd/data/omarabu/001/gonzalo/flashscenic_2/flashscenic/experiments/logs/%x_%j.err
#SBATCH -t 0-10:00:00
#SBATCH -c 16
#SBATCH --mem 256G
#SBATCH -p mit_preemptable
#SBATCH --gres=gpu:a100:1

set -euo pipefail

SCRIPT_DIR="/orcd/data/omarabu/001/gonzalo/flashscenic_2/flashscenic/experiments/robustness"
cd "$SCRIPT_DIR"

mkdir -p /orcd/data/omarabu/001/gonzalo/flashscenic_2/flashscenic/experiments/logs

source /orcd/data/omarabu/001/gonzalo/aging_gde/bin/activate

# ── Parse optional overrides ─────────────────────────────────────────────────
N_RUNS=30
N_STEPS=1000
DEVICE="cuda"
FORCE_PREPROCESS=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --n_runs)           N_RUNS="$2";       shift 2 ;;
        --n_steps)          N_STEPS="$2";      shift 2 ;;
        --device)           DEVICE="$2";       shift 2 ;;
        --force-preprocess) FORCE_PREPROCESS=1; shift 1 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Read paths from config.py ─────────────────────────────────────────────────
echo "[config] Reading paths from config.py..."
read -r PREPROCESSED_PATH GRN_DATA_PATH RESULTS_DIR SPECIES <<< "$(python - <<'PYEOF'
import sys; sys.path.insert(0, "/orcd/data/omarabu/001/gonzalo/flashscenic_2/flashscenic")
from experiments.robustness.config import (
    PREPROCESSED_PATH, GRN_DATA_PATH, RESULTS_DIR,
    DATASETS
)
species = DATASETS["als_spinal_cord"]["species"]
print(PREPROCESSED_PATH, GRN_DATA_PATH, RESULTS_DIR, species)
PYEOF
)"

# ── Timestamped output directory for this run ────────────────────────────────
TIMESTAMP="$(date +"%Y-%m-%d_%H-%M-%S")"
RESULTS_DIR="/orcd/data/omarabu/001/gonzalo/flashscenic_2/flashscenic/experiments/robustness/results"
FIGURES_DIR="${RESULTS_DIR}/${TIMESTAMP}/figures"
mkdir -p "$FIGURES_DIR"

echo "  PREPROCESSED_PATH : $PREPROCESSED_PATH"
echo "  GRN_DATA_PATH     : $GRN_DATA_PATH"
echo "  RUN_DIR           : ${RESULTS_DIR}/${TIMESTAMP}"
echo "  FIGURES_DIR       : $FIGURES_DIR"
echo "  SPECIES           : $SPECIES"
echo "  N_RUNS            : $N_RUNS"
echo "  N_STEPS           : $N_STEPS"
echo "  DEVICE            : $DEVICE"
echo ""

# ── Step 1: Preprocessing (run only if outputs are missing) ──────────────────
if [[ $FORCE_PREPROCESS -eq 1 ]]; then
    echo "[1/3] --force-preprocess set: removing cached preprocessed files..."
    rm -f "$PREPROCESSED_PATH" "$GRN_DATA_PATH"
fi

if [[ -f "$PREPROCESSED_PATH" && -f "$GRN_DATA_PATH" ]]; then
    echo "[1/3] Preprocessing outputs already exist — skipping."
    echo "      $PREPROCESSED_PATH"
    echo "      $GRN_DATA_PATH"
else
    echo "[1/3] Running preprocessing..."
    python 01_preprocessing.py
    echo "      Done."
fi
echo ""

# ── Step 2: Experiment 1 — Bland-Altman adjacency deviation ──────────────────
echo "[2/3] Experiment 1: Bland-Altman adjacency deviation..."
python exp1_bland_altman.py \
    --h5ad "$GRN_DATA_PATH" \
    --output_dir "$FIGURES_DIR" \
    --n_runs "$N_RUNS" \
    --n_steps "$N_STEPS" \
    --device "$DEVICE"
echo "      Done. → $FIGURES_DIR/exp1_bland_altman.pdf"
echo ""

# ── Step 3: Experiments 2 & 3 — Regulon convergence + AUCell stability ───────
echo "[3/3] Experiments 2 & 3: Regulon set convergence + AUCell stability..."
python exp2_3_combined.py \
    --h5ad "$GRN_DATA_PATH" \
    --output_dir "$FIGURES_DIR" \
    --n_runs "$N_RUNS" \
    --n_steps "$N_STEPS" \
    --species "$SPECIES" \
    --device "$DEVICE"
echo "      Done. → $FIGURES_DIR/exp2_jaccard_convergence.pdf"
echo "             → $FIGURES_DIR/exp3_aucell_stability.pdf"
echo "             → $FIGURES_DIR/exp_regulon_counts.pdf"
echo ""

echo "============================================================"
echo "All experiments complete. Figures saved to:"
echo "  $FIGURES_DIR"
echo "============================================================"
