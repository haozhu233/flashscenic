# Multi-Run Ensemble

`multi_run_flashscenic()` runs RegDiffusion `n_runs` times, averages the resulting adjacency matrices into a consensus (optionally masking edges with high run-to-run variability), and then runs the rest of the pipeline once on that stable consensus.

## Why

RegDiffusion's GRN inference is stochastic — running it multiple times on the same expression data produces adjacency matrices that agree on strong edges but disagree on weaker ones. A single `run_flashscenic()` call takes whatever one stochastic draw happens to give you. `multi_run_flashscenic()` averages several draws first, so the regulons that come out the other end are less sensitive to any one run's noise.

This is architecturally different from post-pruning frequency aggregation (running the full pipeline `n_runs` times and majority-voting on the final regulons): it averages continuous edge weights *before* any binary thresholding occurs, so a weak-but-consistent edge and a strong-but-noisy edge are distinguished by their variability, not just their final presence or absence.

## Overview

```
n_runs x RegDiffusion GRN Inference
    │
    ▼
Average adjacency matrices → consensus          ← cv_threshold
    │
    ▼
Steps 2-5 (TF filtering → AUCell), run once      ← same as run_flashscenic()
    │
    ▼
Result dict (auc_scores, regulons, ..., adj_mean, adj_cv)
```

## Basic Usage

```python
import flashscenic as fs

result = fs.multi_run_flashscenic(
    exp_matrix, gene_names, species='human',
    n_runs=15,          # number of independent RegDiffusion runs to average (default)
    cv_threshold=None,  # default: no masking, use the raw mean
    seed=42,            # reduces (does not guarantee-eliminate) GPU run-to-run variation
)

# Same result fields as run_flashscenic(), plus multi-run diagnostics:
auc_scores = result['auc_scores']            # (n_cells, n_regulons)
adj_mean = result['adj_mean']                # (n_genes, n_genes) consensus adjacency
adj_cv = result['adj_cv']                    # (n_genes, n_genes) per-edge coefficient of variation
n_filtered = result['n_edges_cv_filtered']   # edges removed by the CV mask (0 unless cv_threshold is set)
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_runs` | `15` | Number of independent RegDiffusion GRN inference runs to average |
| `cv_threshold` | `None` | Coefficient of variation threshold for edge masking. Edges with std/&#124;mean&#124; &ge; this are zeroed in the consensus. `None` disables masking (raw mean only) |
| `return_adj_matrices` | `False` | If `True`, includes all `n_runs` raw adjacency matrices in the result under `'adj_matrices'`. Off by default to save memory on large expression matrices |
| `seeds` | `None` | Per-run seeds (list of length `n_runs`), passed to `torch.manual_seed` before each RegDiffusion training. Takes priority over `seed` if both are set |
| `seed` | `None` | Single seed for the whole ensemble. If `seeds` is not set, `n_runs` per-run seeds are derived from it (via `numpy.random.SeedSequence(seed).spawn(n_runs)`) and used for the ensemble; also forwarded to the downstream `run_flashscenic()` call for AUCell's tie-breaking noise |

All remaining parameters (the `grn_`, `module_`, `pruning_`, `annotation_`, `aucell_`-prefixed ones, plus `device`/`verbose`) match `run_flashscenic()` and are forwarded unchanged to the downstream pipeline (Steps 2-5 only — `grn_n_steps` still controls each ensemble member's training length).

### Choosing `cv_threshold`

In practice, thresholds `>= 1.0` rarely filter anything: edges that survive `grn_sparsity_threshold` (Step 2) already tend to have low run-to-run variability, so CV masking only has a visible effect around `0.25` or lower.

```{important}
Robustness testing against an independent TF-target database (CollecTRI) did not find that strict CV filtering improves downstream regulon quality — stricter thresholds performed no better, and sometimes worse, than the unfiltered mean (`cv_threshold=None`). Treat this as an available knob for your own experimentation rather than something to enable by default; `None` is a reasonable starting point and the current default.
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
| `adj_mean` | `(n_genes, n_genes)` — the consensus adjacency matrix used for Steps 2-5 |
| `adj_cv` | `(n_genes, n_genes)` — per-edge coefficient of variation (std/\|mean\|) across the `n_runs` draws |
| `adj_matrices` | List of `n_runs` raw adjacency matrices, or `None` if `return_adj_matrices=False` |
| `n_runs` | The `n_runs` value used |
| `cv_threshold` | The `cv_threshold` value used |
| `n_edges_cv_filtered` | Number of edges zeroed by the CV mask (`0` if `cv_threshold=None`) |
