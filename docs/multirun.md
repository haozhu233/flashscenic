# Multi-Run Ensemble

`multi_run_flashscenic()` runs RegDiffusion `n_runs` times and aggregates the runs into a consensus set of regulons, using one of two strategies controlled by `aggregation_method`.

## Why

RegDiffusion's GRN inference is stochastic — running it multiple times on the same expression data produces adjacency matrices that agree on strong edges but disagree on weaker ones. A single `run_flashscenic()` call takes whatever one stochastic draw happens to give you. `multi_run_flashscenic()` combines several draws first, so the regulons that come out the other end are less sensitive to any one run's noise.

## Aggregation methods

| | `aggregation_method='frequency'` (default) | `aggregation_method='mean_adjacency'` |
|---|---|---|
| What it does | Runs the *full* pipeline (module selection through cisTarget pruning) `n_runs` times independently, keeps a (TF, gene) edge if it survives pruning in `>= frequency_threshold` fraction of runs, drops TFs with fewer than `min_genes_filter` retained targets, then runs AUCell once on the consensus | Averages the `n_runs` adjacency matrices into a continuous consensus (optionally masking high-CV edges), then runs the rest of the pipeline (module selection → AUCell) once on that consensus |
| Cost | `n_runs` full pipeline passes | `n_runs` RegDiffusion-only passes + 1 full downstream pass |
| Regulon quality | Validated (against CollecTRI) to have the best precision and recall of every strategy tested, including every `mean_adjacency` variant | Consistently lower precision/recall than frequency aggregation in the same validation — see `experiments/robustness/` |

Because the downstream pipeline runs once instead of `n_runs` times, `mean_adjacency` is meaningfully cheaper. It remains available for cases where that cost matters more than the regulon-quality difference, but `frequency` is the recommended default.

## Overview

```
n_runs x RegDiffusion GRN Inference
    │
    ├── aggregation_method='frequency' (default) ─────────────────────────
    │       n_runs x (TF filtering → modules → cisTarget pruning)
    │       Vote: keep (TF,gene) if freq >= frequency_threshold   ← frequency_threshold
    │       Drop TFs with < min_genes_filter retained targets     ← min_genes_filter
    │       AUCell, run once on the consensus regulons
    │
    └── aggregation_method='mean_adjacency' ───────────────────────────────
            Average adjacency matrices → consensus                ← cv_threshold
            Steps 2-5 (TF filtering → AUCell), run once            ← same as run_flashscenic()

Result dict (auc_scores, regulons, ..., adj_mean, adj_cv)
```

## Basic Usage

```python
import flashscenic as fs

# Default: frequency aggregation
result = fs.multi_run_flashscenic(
    exp_matrix, gene_names, species='human',
    n_runs=15,                 # number of independent runs to aggregate (default)
    aggregation_method='frequency',  # default
    frequency_threshold=0.5,   # keep (TF, gene) pairs surviving pruning in >=50% of runs (default)
    min_genes_filter=5,        # drop TFs with fewer than 5 retained target genes (default)
    seed=42,                   # reduces (does not guarantee-eliminate) GPU run-to-run variation
)

# Cheaper alternative: mean-adjacency aggregation
result_mean = fs.multi_run_flashscenic(
    exp_matrix, gene_names, species='human',
    n_runs=15,
    aggregation_method='mean_adjacency',
    cv_threshold=None,         # default: no masking, use the raw mean
    min_genes_filter=5,        # still applied, composed with pruning_min_genes
)

# Same result fields under both methods, plus multi-run diagnostics:
auc_scores = result['auc_scores']            # (n_cells, n_regulons)
adj_mean = result['adj_mean']                # (n_genes, n_genes) mean adjacency across runs
adj_cv = result['adj_cv']                    # (n_genes, n_genes) per-edge coefficient of variation
tf_scores = result['tf_frequency_scores']    # per-TF mean pair-frequency (None under mean_adjacency)
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_runs` | `15` | Number of independent RegDiffusion GRN inference runs |
| `aggregation_method` | `'frequency'` | `'frequency'` or `'mean_adjacency'` — see above |
| `frequency_threshold` | `0.5` | Fraction of runs (0–1) a (TF, gene) pair must survive cisTarget pruning in to be kept. Only used when `aggregation_method='frequency'` |
| `min_genes_filter` | `5` | Minimum target genes a regulon must retain. Applies to both methods — see [Choosing `min_genes_filter`](#choosing-min_genes_filter) below |
| `cv_threshold` | `None` | Coefficient of variation threshold for edge masking. Edges with std/&#124;mean&#124; &ge; this are zeroed in the consensus. Only used when `aggregation_method='mean_adjacency'`; a warning is raised if set together with `aggregation_method='frequency'`, since it would otherwise silently have no effect |
| `return_adj_matrices` | `False` | If `True`, includes all `n_runs` raw adjacency matrices in the result under `'adj_matrices'`. Off by default to save memory on large expression matrices |
| `seeds` | `None` | Per-run seeds (list of length `n_runs`), passed to `torch.manual_seed` before each RegDiffusion training. Takes priority over `seed` if both are set |
| `seed` | `None` | Single seed for the whole ensemble. If `seeds` is not set, `n_runs` per-run seeds are derived from it (via `numpy.random.SeedSequence(seed).spawn(n_runs)`) and used for the ensemble; also forwarded to every downstream `run_flashscenic()` call for AUCell's tie-breaking noise |

All remaining parameters (the `grn_`, `module_`, `pruning_`, `annotation_`, `aucell_`-prefixed ones, plus `device`/`verbose`) match `run_flashscenic()` and are forwarded unchanged. `grn_n_steps` still controls each ensemble member's training length under both aggregation methods.

### Choosing `min_genes_filter`

- `aggregation_method='frequency'`: applied *after* frequency aggregation — any TF whose consensus target count falls below `min_genes_filter` is dropped entirely.
- `aggregation_method='mean_adjacency'`: composed with `pruning_min_genes` on the single downstream `run_flashscenic` call as `max(pruning_min_genes, min_genes_filter)` — whichever floor is stricter wins, so an explicit `pruning_min_genes` you pass is never silently overridden by a lower `min_genes_filter`.

### Choosing `cv_threshold` (mean_adjacency only)

In practice, thresholds `>= 1.0` rarely filter anything: edges that survive `grn_sparsity_threshold` (Step 2) already tend to have low run-to-run variability, so CV masking only has a visible effect around `0.25` or lower.

```{important}
Robustness testing against an independent TF-target database (CollecTRI) did not find that strict CV filtering improves downstream regulon quality — stricter thresholds performed no better, and sometimes worse, than the unfiltered mean (`cv_threshold=None`). Treat this as an available knob for your own experimentation rather than something to enable by default.
```

**Tuning tips:**
- Inspect `result['adj_cv']` and `result['n_edges_cv_filtered']` before committing to a threshold — at common values (`>= 1.0`) filtering may simply be a no-op on your data
- If you do enable filtering, validate the resulting regulons against an independent source (e.g. a curated TF-target database) rather than assuming stricter is better
- Fewer than 2 runs makes `adj_cv` trivially 0 everywhere (no variability to measure), so `cv_threshold` becomes a no-op regardless of threshold — a warning is raised in that case

## Seeding and Reproducibility

Pass `seed` for an easy single-value ensemble seed, or `seeds` (a list) if you need explicit control over each run's seed individually:

```python
# Single seed, expanded internally into n_runs distinct per-run seeds
result = fs.multi_run_flashscenic(exp_matrix, gene_names, n_runs=15, seed=42)

# Explicit per-run seeds (must have length n_runs)
result = fs.multi_run_flashscenic(
    exp_matrix, gene_names, n_runs=3, seeds=[100, 200, 300],
)
```

```{important}
Seeding reduces but does **not** guarantee-eliminate run-to-run variation on GPU. `torch.manual_seed` controls PyTorch's RNG state, but some CUDA operations are non-deterministic regardless of seed (parallel floating-point reductions don't have a fixed summation order). Two runs with the same seed will be far more similar than two unseeded runs, but are not guaranteed to be bit-identical.
```

## Return Value

In addition to every field `run_flashscenic()` returns (`auc_scores`, `regulon_names`, `regulons`, `regulon_adj`, `parameters`, ...), the result dict includes:

| Field | Description |
|-------|-------------|
| `adj_mean` | `(n_genes, n_genes)` — mean adjacency across the `n_runs` draws (the consensus used for Steps 2-5 under `aggregation_method='mean_adjacency'`; a diagnostic only under `'frequency'`) |
| `adj_cv` | `(n_genes, n_genes)` — per-edge coefficient of variation (std/\|mean\|) across the `n_runs` draws |
| `adj_matrices` | List of `n_runs` raw adjacency matrices, or `None` if `return_adj_matrices=False` |
| `n_runs` | The `n_runs` value used |
| `aggregation_method` | The `aggregation_method` value used |
| `frequency_threshold` | The `frequency_threshold` value used |
| `min_genes_filter` | The `min_genes_filter` value used |
| `cv_threshold` | The `cv_threshold` value used |
| `n_edges_cv_filtered` | Number of edges zeroed by the CV mask (`0` if `cv_threshold=None` or `aggregation_method='frequency'`) |
| `tf_frequency_scores` | Dict mapping each consensus TF to its mean pair-frequency among retained genes, or `None` under `aggregation_method='mean_adjacency'` |
