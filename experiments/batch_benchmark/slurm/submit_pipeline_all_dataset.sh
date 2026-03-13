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
# To skip steps, edit SKIP_STEPS below.
# Valid step IDs: 00 01 02A 02B 02C 02D 03 04 05 05B 06
# Per-dataset steps (03, 04, 05, 05B) skip all three dataset jobs for that step.
# Example: SKIP_STEPS=(00 01 02A 02B 02C 02D) — resume from metrics onwards
#
# Prerequisites (run on login node first):
#   bash slurm/00_download_data.sh
#   bash slurm/00_prefetch_resources.sh

set -euo pipefail

BENCH_DIR=/cluster/scratch/gcardenal/flashscenic/experiments/batch_benchmark
LOGS="$BENCH_DIR/logs"
mkdir -p "$LOGS/submit_pipeline"

# ---------------------------------------------------------------------------
# Steps to skip — add step IDs to this array
# ---------------------------------------------------------------------------
SKIP_STEPS=(00)   # e.g. (00 01 02A 02B 02C 02D 03 04 05 05B 06)

# ---------------------------------------------------------------------------
# Datasets to skip — set any per-dataset job var to "skip" to exclude it
# from all steps. Useful when a dataset is commented out in config.py.
# ---------------------------------------------------------------------------
JOB_03_AD=skip; JOB_04_AD=skip; JOB_05_AD=skip; JOB_05B_AD=skip
# JOB_03_MOTOR=skip; JOB_04_MOTOR=skip; JOB_05_MOTOR=skip; JOB_05B_MOTOR=skip
# JOB_03_SPINE=skip; JOB_04_SPINE=skip; JOB_05_SPINE=skip; JOB_05B_SPINE=skip

_is_skipped() {
    local step="$1"
    local s
    for s in "${SKIP_STEPS[@]+"${SKIP_STEPS[@]}"}"; do
        [[ "$s" == "$step" ]] && return 0
    done
    return 1
}

# ---------------------------------------------------------------------------
# Dry-run support
# ---------------------------------------------------------------------------
DRY_RUN=false
for arg in "$@"; do
    [[ "$arg" == "--dry-run" ]] && DRY_RUN=true
done

_submit() {
    # Usage: _submit VARNAME [--dependency=<dep>] [--extra-sbatch-flags...] SCRIPT
    # Any --flag that is not --dependency= is forwarded directly to sbatch.
    # Sets VARNAME to the submitted job ID (or a placeholder in dry-run).
    # If VARNAME is already set to "skip", honour that and do not submit.
    local varname="$1"
    if [[ "${!varname:-}" == "skip" ]]; then
        return 0
    fi
    shift
    local dep_flag=""
    local script=""
    local extra_sbatch_flags=()

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
        elif [[ "$arg" == --* ]]; then
            extra_sbatch_flags+=("$arg")
        else
            script="$arg"
        fi
    done

    local script_name
    script_name=$(basename "$script" .sh)

    if $DRY_RUN; then
        local fake_id="DRY_${script_name}"
        local flags_str="${extra_sbatch_flags[*]:-}"
        printf '  [dry-run] Would submit: sbatch %s %s %s\n' \
            "${flags_str}" "${dep_flag:-(no dependency)}" "$script"
        printf -v "$varname" '%s' "$fake_id"
    else
        local job_id
        if [[ -n "$dep_flag" ]]; then
            job_id=$(sbatch --parsable "${extra_sbatch_flags[@]+"${extra_sbatch_flags[@]}"}" "$dep_flag" "$script")
        else
            job_id=$(sbatch --parsable "${extra_sbatch_flags[@]+"${extra_sbatch_flags[@]}"}" "$script")
        fi
        printf -v "$varname" '%s' "$job_id"
    fi
}

echo "=== Submitting batch benchmark pipeline ==="
echo "Bench dir: $BENCH_DIR"
$DRY_RUN && echo "[DRY RUN — no jobs will actually be submitted]"
[[ ${#SKIP_STEPS[@]} -gt 0 ]] && echo "Skipping steps: ${SKIP_STEPS[*]}"
echo ""

# ---------------------------------------------------------------------------
# Step 00 — Validate data
# ---------------------------------------------------------------------------
if _is_skipped 00; then
    echo "[skip] 00_validate"
    JOB_00=skip
else
    _submit JOB_00 "$BENCH_DIR/slurm/run_00_validate.sh"
    echo "Submitted 00_validate        → job $JOB_00"
fi

# ---------------------------------------------------------------------------
# Step 01 — Preprocess (waits for 00 if not skipped)
# ---------------------------------------------------------------------------
if _is_skipped 01; then
    echo "[skip] 01_preprocess"
    JOB_01=skip
else
    _submit JOB_01 --dependency=afterok:$JOB_00 "$BENCH_DIR/slurm/run_01_preprocess.sh"
    echo "Submitted 01_preprocess      → job $JOB_01"
fi

# ---------------------------------------------------------------------------
# Steps 02a, 02b, 02c — Integration methods (parallel after 01)
# ---------------------------------------------------------------------------
if _is_skipped 02A; then
    echo "[skip] 02a_baselines"
    JOB_02A=skip
else
    _submit JOB_02A --dependency=afterok:$JOB_01 "$BENCH_DIR/slurm/run_02a_baselines.sh"
    echo "Submitted 02a_baselines      → job $JOB_02A"
fi

if _is_skipped 02B; then
    echo "[skip] 02b_scvi"
    JOB_02B=skip
else
    _submit JOB_02B --dependency=afterok:$JOB_01 "$BENCH_DIR/slurm/run_02b_scvi.sh"
    echo "Submitted 02b_scvi           → job $JOB_02B"
fi

if _is_skipped 02C; then
    echo "[skip] 02c_flashscenic"
    JOB_02C=skip
else
    _submit JOB_02C --dependency=afterok:$JOB_01 "$BENCH_DIR/slurm/run_02c_flashscenic.sh"
    echo "Submitted 02c_flashscenic    → job $JOB_02C"
fi

# ---------------------------------------------------------------------------
# Step 02d — Merge all embeddings into h5ad (waits for ALL 02x, runs once)
# ---------------------------------------------------------------------------
if _is_skipped 02D; then
    echo "[skip] 02d_merge"
    JOB_02D=skip
else
    _submit JOB_02D --dependency=afterok:${JOB_02A}:${JOB_02B}:${JOB_02C} "$BENCH_DIR/slurm/run_02d_merge.sh"
    echo "Submitted 02d_merge          → job $JOB_02D"
fi

# ---------------------------------------------------------------------------
# Steps 03 — scIB metrics, one job per dataset (parallel after 02d)
# ---------------------------------------------------------------------------
mkdir -p "$LOGS/03_metrics" "$LOGS/04_ml_predictor" "$LOGS/05_visualize" "$LOGS/05b_rss"

if _is_skipped 03; then
    echo "[skip] 03_metrics (all datasets)"
    JOB_03_IMMUNE=skip; JOB_03_PANC=skip; JOB_03_AD=skip; JOB_03_INHIB=skip
else
    _submit JOB_03_IMMUNE \
        --job-name=03_metrics_immune \
        --output="$LOGS/03_metrics/03_metrics_immune_%j.out" \
        --export=ALL,DATASET=immune_human \
        --dependency=afterok:${JOB_02D} \
        "$BENCH_DIR/slurm/run_03_metrics.sh"
    echo "Submitted 03_metrics_immune  → job $JOB_03_IMMUNE"

    _submit JOB_03_PANC \
        --job-name=03_metrics_pancreas \
        --output="$LOGS/03_metrics/03_metrics_pancreas_%j.out" \
        --export=ALL,DATASET=pancreas \
        --dependency=afterok:${JOB_02D} \
        "$BENCH_DIR/slurm/run_03_metrics.sh"
    echo "Submitted 03_metrics_pancreas→ job $JOB_03_PANC"

    _submit JOB_03_AD \
        --job-name=03_metrics_ad \
        --output="$LOGS/03_metrics/03_metrics_ad_%j.out" \
        --export=ALL,DATASET=ad_neurons \
        --dependency=afterok:${JOB_02D} \
        "$BENCH_DIR/slurm/run_03_metrics.sh"
    echo "Submitted 03_metrics_ad      → job $JOB_03_AD"

    _submit JOB_03_INHIB \
        --job-name=03_metrics_inhib \
        --output="$LOGS/03_metrics/03_metrics_inhib_%j.out" \
        --export=ALL,DATASET=ad_inhibitory \
        --dependency=afterok:${JOB_02D} \
        "$BENCH_DIR/slurm/run_03_metrics.sh"
    echo "Submitted 03_metrics_inhib   → job $JOB_03_INHIB"

    _submit JOB_03_MOTOR \
        --job-name=03_metrics_motor \
        --output="$LOGS/03_metrics/03_metrics_motor_%j.out" \
        --export=ALL,DATASET=als_motor_cortex \
        --dependency=afterok:${JOB_02D} \
        "$BENCH_DIR/slurm/run_03_metrics.sh"
    echo "Submitted 03_metrics_motor   → job $JOB_03_MOTOR"

    _submit JOB_03_SPINE \
        --job-name=03_metrics_spine \
        --output="$LOGS/03_metrics/03_metrics_spine_%j.out" \
        --export=ALL,DATASET=als_spinal_cord \
        --dependency=afterok:${JOB_02D} \
        "$BENCH_DIR/slurm/run_03_metrics.sh"
    echo "Submitted 03_metrics_spine   → job $JOB_03_SPINE"
fi

# ---------------------------------------------------------------------------
# Steps 04 — ML predictor, one job per dataset (parallel after 02d)
# ---------------------------------------------------------------------------
if _is_skipped 04; then
    echo "[skip] 04_ml_predictor (all datasets)"
    JOB_04_IMMUNE=skip; JOB_04_PANC=skip; JOB_04_AD=skip; JOB_04_INHIB=skip
else
    _submit JOB_04_IMMUNE \
        --job-name=04_ml_immune \
        --output="$LOGS/04_ml_predictor/04_ml_immune_%j.out" \
        --export=ALL,DATASET=immune_human \
        --dependency=afterok:${JOB_02D} \
        "$BENCH_DIR/slurm/run_04_ml_predictor.sh"
    echo "Submitted 04_ml_immune       → job $JOB_04_IMMUNE"

    _submit JOB_04_PANC \
        --job-name=04_ml_pancreas \
        --output="$LOGS/04_ml_predictor/04_ml_pancreas_%j.out" \
        --export=ALL,DATASET=pancreas \
        --dependency=afterok:${JOB_02D} \
        "$BENCH_DIR/slurm/run_04_ml_predictor.sh"
    echo "Submitted 04_ml_pancreas     → job $JOB_04_PANC"

    _submit JOB_04_AD \
        --job-name=04_ml_ad \
        --output="$LOGS/04_ml_predictor/04_ml_ad_%j.out" \
        --export=ALL,DATASET=ad_neurons \
        --dependency=afterok:${JOB_02D} \
        "$BENCH_DIR/slurm/run_04_ml_predictor.sh"
    echo "Submitted 04_ml_ad           → job $JOB_04_AD"

    _submit JOB_04_INHIB \
        --job-name=04_ml_inhib \
        --output="$LOGS/04_ml_predictor/04_ml_inhib_%j.out" \
        --export=ALL,DATASET=ad_inhibitory,TASKS="sex age" \
        --dependency=afterok:${JOB_02D} \
        "$BENCH_DIR/slurm/run_04_ml_predictor.sh"
    echo "Submitted 04_ml_inhib        → job $JOB_04_INHIB"

    _submit JOB_04_MOTOR \
        --job-name=04_ml_motor \
        --output="$LOGS/04_ml_predictor/04_ml_motor_%j.out" \
        --export=ALL,DATASET=als_motor_cortex \
        --dependency=afterok:${JOB_02D} \
        "$BENCH_DIR/slurm/run_04_ml_predictor.sh"
    echo "Submitted 04_ml_motor        → job $JOB_04_MOTOR"

    _submit JOB_04_SPINE \
        --job-name=04_ml_spine \
        --output="$LOGS/04_ml_predictor/04_ml_spine_%j.out" \
        --export=ALL,DATASET=als_spinal_cord \
        --dependency=afterok:${JOB_02D} \
        "$BENCH_DIR/slurm/run_04_ml_predictor.sh"
    echo "Submitted 04_ml_spine        → job $JOB_04_SPINE"
fi

# ---------------------------------------------------------------------------
# Steps 05 — Visualize, one job per dataset (each waits for its own 03+04)
# ---------------------------------------------------------------------------
if _is_skipped 05; then
    echo "[skip] 05_visualize (all datasets)"
    JOB_05_IMMUNE=skip; JOB_05_PANC=skip; JOB_05_AD=skip; JOB_05_INHIB=skip
else
    _submit JOB_05_IMMUNE \
        --job-name=05_visualize_immune \
        --output="$LOGS/05_visualize/05_visualize_immune_%j.out" \
        --export=ALL,DATASET=immune_human \
        --dependency=afterok:${JOB_03_IMMUNE}:${JOB_04_IMMUNE} \
        "$BENCH_DIR/slurm/run_05_visualize.sh"
    echo "Submitted 05_visualize_immune→ job $JOB_05_IMMUNE"

    _submit JOB_05_PANC \
        --job-name=05_visualize_pancreas \
        --output="$LOGS/05_visualize/05_visualize_pancreas_%j.out" \
        --export=ALL,DATASET=pancreas \
        --dependency=afterok:${JOB_03_PANC}:${JOB_04_PANC} \
        "$BENCH_DIR/slurm/run_05_visualize.sh"
    echo "Submitted 05_visualize_panc  → job $JOB_05_PANC"

    _submit JOB_05_AD \
        --job-name=05_visualize_ad \
        --output="$LOGS/05_visualize/05_visualize_ad_%j.out" \
        --export=ALL,DATASET=ad_neurons \
        --dependency=afterok:${JOB_03_AD}:${JOB_04_AD} \
        "$BENCH_DIR/slurm/run_05_visualize.sh"
    echo "Submitted 05_visualize_ad    → job $JOB_05_AD"

    _submit JOB_05_INHIB \
        --job-name=05_visualize_inhib \
        --output="$LOGS/05_visualize/05_visualize_inhib_%j.out" \
        --export=ALL,DATASET=ad_inhibitory \
        --dependency=afterok:${JOB_03_INHIB}:${JOB_04_INHIB} \
        "$BENCH_DIR/slurm/run_05_visualize.sh"
    echo "Submitted 05_visualize_inhib → job $JOB_05_INHIB"

    _submit JOB_05_MOTOR \
        --job-name=05_visualize_motor \
        --output="$LOGS/05_visualize/05_visualize_motor_%j.out" \
        --export=ALL,DATASET=als_motor_cortex \
        --dependency=afterok:${JOB_03_MOTOR}:${JOB_04_MOTOR} \
        "$BENCH_DIR/slurm/run_05_visualize.sh"
    echo "Submitted 05_visualize_motor → job $JOB_05_MOTOR"

    _submit JOB_05_SPINE \
        --job-name=05_visualize_spine \
        --output="$LOGS/05_visualize/05_visualize_spine_%j.out" \
        --export=ALL,DATASET=als_spinal_cord \
        --dependency=afterok:${JOB_03_SPINE}:${JOB_04_SPINE} \
        "$BENCH_DIR/slurm/run_05_visualize.sh"
    echo "Submitted 05_visualize_spine → job $JOB_05_SPINE"
fi

# ---------------------------------------------------------------------------
# Steps 05b — RSS, one job per dataset (each waits for its own 05)
# ---------------------------------------------------------------------------
if _is_skipped 05B; then
    echo "[skip] 05b_rss (all datasets)"
    JOB_05B_IMMUNE=skip; JOB_05B_PANC=skip; JOB_05B_AD=skip; JOB_05B_INHIB=skip
else
    _submit JOB_05B_IMMUNE \
        --job-name=05b_rss_immune \
        --output="$LOGS/05b_rss/05b_rss_immune_%j.out" \
        --export=ALL,DATASET=immune_human \
        --dependency=afterok:${JOB_05_IMMUNE} \
        "$BENCH_DIR/slurm/run_05b_rss.sh"
    echo "Submitted 05b_rss_immune     → job $JOB_05B_IMMUNE"

    _submit JOB_05B_PANC \
        --job-name=05b_rss_pancreas \
        --output="$LOGS/05b_rss/05b_rss_pancreas_%j.out" \
        --export=ALL,DATASET=pancreas \
        --dependency=afterok:${JOB_05_PANC} \
        "$BENCH_DIR/slurm/run_05b_rss.sh"
    echo "Submitted 05b_rss_pancreas   → job $JOB_05B_PANC"

    _submit JOB_05B_AD \
        --job-name=05b_rss_ad \
        --output="$LOGS/05b_rss/05b_rss_ad_%j.out" \
        --export=ALL,DATASET=ad_neurons \
        --dependency=afterok:${JOB_05_AD} \
        "$BENCH_DIR/slurm/run_05b_rss.sh"
    echo "Submitted 05b_rss_ad         → job $JOB_05B_AD"

    _submit JOB_05B_INHIB \
        --job-name=05b_rss_inhib \
        --output="$LOGS/05b_rss/05b_rss_inhib_%j.out" \
        --export=ALL,DATASET=ad_inhibitory \
        --dependency=afterok:${JOB_05_INHIB} \
        "$BENCH_DIR/slurm/run_05b_rss.sh"
    echo "Submitted 05b_rss_inhib      → job $JOB_05B_INHIB"

    _submit JOB_05B_MOTOR \
        --job-name=05b_rss_motor \
        --output="$LOGS/05b_rss/05b_rss_motor_%j.out" \
        --export=ALL,DATASET=als_motor_cortex \
        --dependency=afterok:${JOB_05_MOTOR} \
        "$BENCH_DIR/slurm/run_05b_rss.sh"
    echo "Submitted 05b_rss_motor      → job $JOB_05B_MOTOR"

    _submit JOB_05B_SPINE \
        --job-name=05b_rss_spine \
        --output="$LOGS/05b_rss/05b_rss_spine_%j.out" \
        --export=ALL,DATASET=als_spinal_cord \
        --dependency=afterok:${JOB_05_SPINE} \
        "$BENCH_DIR/slurm/run_05b_rss.sh"
    echo "Submitted 05b_rss_spine      → job $JOB_05B_SPINE"
fi

# ---------------------------------------------------------------------------
# Step 06 — Summarize (waits for ALL 05b jobs)
# ---------------------------------------------------------------------------
if _is_skipped 06; then
    echo "[skip] 06_summarize"
    JOB_06=skip
else
    _submit JOB_06 \
        --dependency=afterok:${JOB_05B_IMMUNE}:${JOB_05B_PANC}:${JOB_05B_AD}:${JOB_05B_INHIB}:${JOB_05B_MOTOR}:${JOB_05B_SPINE} \
        "$BENCH_DIR/slurm/run_06_summarize.sh"
    echo "Submitted 06_summarize       → job $JOB_06"
fi

echo ""
echo "=== All jobs submitted ==="
echo ""
echo "Job chain:"
echo "  $JOB_00              00_validate"
echo "  $JOB_01              01_preprocess"
echo "  $JOB_02A             02a_baselines  ┐"
echo "  $JOB_02B             02b_scvi       ├── parallel"
echo "  $JOB_02C             02c_flashscenic┘"
echo "  $JOB_02D             02d_merge"
echo "  $JOB_03_IMMUNE       03_metrics_immune   ┐"
echo "  $JOB_03_PANC         03_metrics_pancreas │"
echo "  $JOB_03_AD           03_metrics_ad       ├── parallel"
echo "  $JOB_03_INHIB        03_metrics_inhib    │"
echo "  $JOB_03_MOTOR        03_metrics_motor    │"
echo "  $JOB_03_SPINE        03_metrics_spine    ┘"
echo "  $JOB_04_IMMUNE       04_ml_immune        ┐"
echo "  $JOB_04_PANC         04_ml_pancreas      │"
echo "  $JOB_04_AD           04_ml_ad            ├── parallel"
echo "  $JOB_04_INHIB        04_ml_inhib         │"
echo "  $JOB_04_MOTOR        04_ml_motor         │"
echo "  $JOB_04_SPINE        04_ml_spine         ┘"
echo "  $JOB_05_IMMUNE       05_visualize_immune ┐ (waits for 03+04 per dataset)"
echo "  $JOB_05_PANC         05_visualize_panc   │"
echo "  $JOB_05_AD           05_visualize_ad     ├── parallel"
echo "  $JOB_05_INHIB        05_visualize_inhib  │"
echo "  $JOB_05_MOTOR        05_visualize_motor  │"
echo "  $JOB_05_SPINE        05_visualize_spine  ┘"
echo "  $JOB_05B_IMMUNE      05b_rss_immune      ┐"
echo "  $JOB_05B_PANC        05b_rss_pancreas    │"
echo "  $JOB_05B_AD          05b_rss_ad          ├── parallel"
echo "  $JOB_05B_INHIB       05b_rss_inhib       │"
echo "  $JOB_05B_MOTOR       05b_rss_motor       │"
echo "  $JOB_05B_SPINE       05b_rss_spine       ┘"
echo "  $JOB_06              06_summarize"

if ! $DRY_RUN; then
    echo ""
    echo "Monitor with:"
    echo "  squeue -u \$USER"
    JOB_LIST=$(echo "${JOB_00} ${JOB_01} ${JOB_02A} ${JOB_02B} ${JOB_02C} ${JOB_02D} \
                     ${JOB_03_IMMUNE} ${JOB_03_PANC} ${JOB_03_AD} ${JOB_03_INHIB} ${JOB_03_MOTOR} ${JOB_03_SPINE} \
                     ${JOB_04_IMMUNE} ${JOB_04_PANC} ${JOB_04_AD} ${JOB_04_INHIB} ${JOB_04_MOTOR} ${JOB_04_SPINE} \
                     ${JOB_05_IMMUNE} ${JOB_05_PANC} ${JOB_05_AD} ${JOB_05_INHIB} ${JOB_05_MOTOR} ${JOB_05_SPINE} \
                     ${JOB_05B_IMMUNE} ${JOB_05B_PANC} ${JOB_05B_AD} ${JOB_05B_INHIB} ${JOB_05B_MOTOR} ${JOB_05B_SPINE} \
                     ${JOB_06}" \
               | tr ' ' '\n' | grep -E '^[0-9]+$' | tr '\n' ',' | sed 's/,$//')
    echo "  squeue -j ${JOB_LIST}"
fi

echo ""
echo "Logs: $BENCH_DIR/logs/<step>/"
echo "Final results: $BENCH_DIR/results/SUMMARY.md"
