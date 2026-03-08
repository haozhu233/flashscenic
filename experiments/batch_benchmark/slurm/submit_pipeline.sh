#!/usr/bin/env bash
#SBATCH --job-name=submit_pipeline
#SBATCH --output=/cluster/scratch/gcardenal/flashscenic/experiments/batch_benchmark/logs/submit_pipeline/%x_%j.out
#SBATCH --time=00:05:00
#SBATCH --mem=256M
#SBATCH --cpus-per-task=1
# submit_pipeline.sh — Submit the full benchmark pipeline with dependency chaining.
#
# Usage:
#   bash submit_pipeline.sh            # submit all jobs
#   bash submit_pipeline.sh --dry-run  # print what would be submitted, no actual submission
#   sbatch submit_pipeline.sh          # submit via SLURM (logs go to logs/submit_pipeline/)
#
# To skip a step: set the variable to "skip" before the step block, e.g.:
#   JOB_00=skip   # validation already done
#
# Prerequisites (run on login node first):
#   bash slurm/00_download_data.sh
#   bash slurm/00_prefetch_resources.sh

set -euo pipefail

BENCH_DIR=/cluster/scratch/gcardenal/flashscenic/experiments/batch_benchmark
mkdir -p "$BENCH_DIR/logs/submit_pipeline"

# ---------------------------------------------------------------------------
# Dry-run support
# ---------------------------------------------------------------------------
DRY_RUN=false
for arg in "$@"; do
    [[ "$arg" == "--dry-run" ]] && DRY_RUN=true
done

_submit() {
    # Usage: _submit VARNAME [--dependency=<dep>] SCRIPT
    # Sets VARNAME to the submitted job ID (or a placeholder in dry-run).
    local varname="$1"; shift
    local dep_flag=""
    local script=""

    for arg in "$@"; do
        if [[ "$arg" == --dependency=* ]]; then
            # Strip any ":skip" entries from the dependency list
            local raw_dep="${arg#--dependency=}"
            local dep_type="${raw_dep%%:*}"       # e.g. "afterok"
            local ids="${raw_dep#*:}"              # e.g. "123:skip:456"
            # Filter out "skip" tokens
            local clean_ids
            clean_ids=$(echo "$ids" | tr ':' '\n' | { grep -v '^skip$' || true; } | tr '\n' ':' | sed 's/:$//')
            if [[ -n "$clean_ids" ]]; then
                dep_flag="--dependency=${dep_type}:${clean_ids}"
            fi
            # If all dependencies were "skip", dep_flag stays empty → no dependency
        else
            script="$arg"
        fi
    done

    local script_name
    script_name=$(basename "$script" .sh)

    if $DRY_RUN; then
        local fake_id="DRY_${script_name}"
        printf '  [dry-run] Would submit: sbatch %s %s\n' "${dep_flag:-(no dependency)}" "$script"
        printf -v "$varname" '%s' "$fake_id"
    else
        local job_id
        if [[ -n "$dep_flag" ]]; then
            job_id=$(sbatch --parsable "$dep_flag" "$script")
        else
            job_id=$(sbatch --parsable "$script")
        fi
        printf -v "$varname" '%s' "$job_id"
    fi
}

echo "=== Submitting batch benchmark pipeline ==="
echo "Bench dir: $BENCH_DIR"
$DRY_RUN && echo "[DRY RUN — no jobs will actually be submitted]"
echo ""

# ---------------------------------------------------------------------------
# Toggle steps here: uncomment the "=skip" line to bypass a step
# ---------------------------------------------------------------------------
JOB_00=skip   # uncomment to skip validation (already done)

# ---------------------------------------------------------------------------
# Step 00 — Validate data
# ---------------------------------------------------------------------------
JOB_00="${JOB_00:-}"   # preserve if already set to "skip" above

if [[ "${JOB_00}" != "skip" ]]; then
    _submit JOB_00 "$BENCH_DIR/slurm/run_00_validate.sh"
    echo "Submitted 00_validate        → job $JOB_00"
else
    echo "[skip] 00_validate"
    JOB_00=skip
fi

# ---------------------------------------------------------------------------
# Step 01 — Preprocess (waits for 00 if not skipped)
# ---------------------------------------------------------------------------
_submit JOB_01 --dependency=afterok:$JOB_00 "$BENCH_DIR/slurm/run_01_preprocess.sh"
echo "Submitted 01_preprocess      → job $JOB_01"

# ---------------------------------------------------------------------------
# Steps 02a, 02b, 02c — Integration methods (parallel after 01)
# ---------------------------------------------------------------------------
_submit JOB_02A --dependency=afterok:$JOB_01 "$BENCH_DIR/slurm/run_02a_baselines.sh"
echo "Submitted 02a_baselines      → job $JOB_02A"

_submit JOB_02B --dependency=afterok:$JOB_01 "$BENCH_DIR/slurm/run_02b_scvi.sh"
echo "Submitted 02b_scvi           → job $JOB_02B"

_submit JOB_02C --dependency=afterok:$JOB_01 "$BENCH_DIR/slurm/run_02c_flashscenic.sh"
echo "Submitted 02c_flashscenic    → job $JOB_02C"

# ---------------------------------------------------------------------------
# Step 02d — Merge all embeddings into h5ad (waits for ALL 02x, runs once)
# ---------------------------------------------------------------------------
_submit JOB_02D --dependency=afterok:${JOB_02A}:${JOB_02B}:${JOB_02C} "$BENCH_DIR/slurm/run_02d_merge.sh"
echo "Submitted 02d_merge          → job $JOB_02D"

# ---------------------------------------------------------------------------
# Steps 03 and 04 — Metrics + ML predictor (both wait for 02d)
# ---------------------------------------------------------------------------
_submit JOB_03 --dependency=afterok:${JOB_02D} "$BENCH_DIR/slurm/run_03_metrics.sh"
echo "Submitted 03_metrics         → job $JOB_03"

_submit JOB_04 --dependency=afterok:${JOB_02D} "$BENCH_DIR/slurm/run_04_ml_predictor.sh"
echo "Submitted 04_ml_predictor    → job $JOB_04"

# ---------------------------------------------------------------------------
# Step 05 — Visualize (waits for 03 and 04)
# ---------------------------------------------------------------------------
_submit JOB_05 --dependency=afterok:${JOB_03}:${JOB_04} "$BENCH_DIR/slurm/run_05_visualize.sh"
echo "Submitted 05_visualize       → job $JOB_05"

# ---------------------------------------------------------------------------
# Step 05b — RSS heatmap from binary AUCell scores (waits for 05)
# ---------------------------------------------------------------------------
_submit JOB_05B --dependency=afterok:${JOB_05} "$BENCH_DIR/slurm/run_05b_rss.sh"
echo "Submitted 05b_rss            → job $JOB_05B"

# ---------------------------------------------------------------------------
# Step 06 — Summarize (waits for 05b)
# ---------------------------------------------------------------------------
_submit JOB_06 --dependency=afterok:${JOB_05B} "$BENCH_DIR/slurm/run_06_summarize.sh"
echo "Submitted 06_summarize       → job $JOB_06"

echo ""
echo "=== All jobs submitted ==="
echo ""
echo "Job chain:"
echo "  $JOB_00   00_validate"
echo "  $JOB_01   01_preprocess"
echo "  $JOB_02A  02a_baselines  ┐"
echo "  $JOB_02B  02b_scvi       ├── parallel"
echo "  $JOB_02C  02c_flashscenic┘"
echo "  $JOB_02D  02d_merge       (serializes h5ad write)"
echo "  $JOB_03   03_metrics      ┐"
echo "  $JOB_04   04_ml_predictor ┘── parallel"
echo "  $JOB_05   05_visualize"
echo "  $JOB_05B  05b_rss"
echo "  $JOB_06   06_summarize"

if ! $DRY_RUN; then
    echo ""
    echo "Monitor with:"
    echo "  squeue -u \$USER"
    JOB_LIST=$(echo "${JOB_00} ${JOB_01} ${JOB_02A} ${JOB_02B} ${JOB_02C} ${JOB_02D} ${JOB_03} ${JOB_04} ${JOB_05} ${JOB_05B} ${JOB_06}" \
               | tr ' ' '\n' | { grep -v '^skip$' || true; } | tr '\n' ',' | sed 's/,$//')
    echo "  squeue -j ${JOB_LIST}"
fi

echo ""
echo "Logs: $BENCH_DIR/logs/<step>/"
echo "Final results: $BENCH_DIR/results/SUMMARY.md"
