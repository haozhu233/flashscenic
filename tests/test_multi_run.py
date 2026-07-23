"""
Tests for multi_run_flashscenic's aggregation logic.

_build_frequency_consensus is pure (no GPU / downloaded resources needed) and
is tested directly. The parameter-validation tests exercise
multi_run_flashscenic itself but only checks that raise/warn before any
GPU work or resource download happens (aggregation_method, frequency_threshold,
seeds, n_runs). The cv_threshold/aggregation_method warning test monkeypatches
the heavy dependencies (resource download, RegDiffusion, run_flashscenic) so it
doesn't need network access or a GPU.
"""

import os
import sys
import warnings

import numpy as np
import pytest

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import flashscenic.data as fs_data
from flashscenic.multi_run import multi_run_flashscenic, _build_frequency_consensus


def _regulon(tf, genes, **overrides):
    reg = {
        'tf': tf, 'genes': genes, 'motif': 'm', 'nes': 1.0,
        'auc': 0.1, 'context': 'c', 'database': 'd',
    }
    reg.update(overrides)
    return reg


class TestBuildFrequencyConsensus:
    GENE_NAMES = [f"g{i}" for i in range(6)]

    def _full_results(self):
        # TF1 pair frequencies across 4 runs: g0=1.0, g1=1.0, g2=0.75, g3=0.5, g4=0.25
        # TF2 pair frequencies across 4 runs: g0=1.0, g1=0.5
        return [
            {'regulons': [_regulon('TF1', ['g0', 'g1', 'g2', 'g3', 'g4']),
                          _regulon('TF2', ['g0', 'g1'])]},
            {'regulons': [_regulon('TF1', ['g0', 'g1', 'g2', 'g3']),
                          _regulon('TF2', ['g0', 'g1'])]},
            {'regulons': [_regulon('TF1', ['g0', 'g1', 'g2']),
                          _regulon('TF2', ['g0'])]},
            {'regulons': [_regulon('TF1', ['g0', 'g1']),
                          _regulon('TF2', ['g0'])]},
        ]

    def test_keeps_pairs_meeting_threshold(self):
        consensus, adj, scores = _build_frequency_consensus(
            self._full_results(), self.GENE_NAMES,
            frequency_threshold=0.5, min_genes_filter=1,
        )
        by_tf = {c['tf']: sorted(c['genes']) for c in consensus}
        assert by_tf == {
            'TF1': ['g0', 'g1', 'g2', 'g3'],  # g4 at 0.25 drops below 0.5
            'TF2': ['g0', 'g1'],
        }
        assert adj.shape == (2, len(self.GENE_NAMES))
        assert scores['TF1'] == pytest.approx((1.0 + 1.0 + 0.75 + 0.5) / 4)
        assert scores['TF2'] == pytest.approx((1.0 + 0.5) / 2)

    def test_min_genes_filter_drops_small_regulons(self):
        # At threshold=0.5: TF1 survives with 4 genes, TF2 with 2 genes.
        consensus, adj, scores = _build_frequency_consensus(
            self._full_results(), self.GENE_NAMES,
            frequency_threshold=0.5, min_genes_filter=3,
        )
        assert [c['tf'] for c in consensus] == ['TF1']
        assert adj.shape == (1, len(self.GENE_NAMES))
        assert 'TF2' not in scores

    def test_stricter_threshold_shrinks_consensus(self):
        consensus, _, _ = _build_frequency_consensus(
            self._full_results(), self.GENE_NAMES,
            frequency_threshold=1.0, min_genes_filter=1,
        )
        by_tf = {c['tf']: sorted(c['genes']) for c in consensus}
        assert by_tf == {'TF1': ['g0', 'g1'], 'TF2': ['g0']}

    def test_empty_consensus_returns_empty(self):
        consensus, adj, scores = _build_frequency_consensus(
            self._full_results(), self.GENE_NAMES,
            frequency_threshold=0.5, min_genes_filter=10,
        )
        assert consensus == []
        assert adj.shape == (0, len(self.GENE_NAMES))
        assert scores == {}

    def test_regulon_dict_schema_matches_cistarget_output(self):
        consensus, _, _ = _build_frequency_consensus(
            self._full_results(), self.GENE_NAMES,
            frequency_threshold=0.5, min_genes_filter=1,
        )
        expected_keys = {
            'name', 'tf', 'motif', 'n_genes', 'genes',
            'context', 'nes', 'auc', 'database',
        }
        for reg in consensus:
            assert set(reg.keys()) == expected_keys
            assert reg['n_genes'] == len(reg['genes'])
            assert reg['name'] == f"{reg['tf']}(+)"


class TestMultiRunValidation:
    """These checks raise before any GPU work or resource download."""

    def _args(self):
        gene_names = [f"g{i}" for i in range(5)]
        exp_matrix = np.zeros((3, 5), dtype=np.float32)
        return exp_matrix, gene_names

    def test_invalid_aggregation_method_raises(self):
        exp_matrix, gene_names = self._args()
        with pytest.raises(ValueError, match="aggregation_method"):
            multi_run_flashscenic(
                exp_matrix, gene_names, aggregation_method="bogus",
            )

    @pytest.mark.parametrize("bad_threshold", [0.0, -0.1, 1.1, 2.0])
    def test_invalid_frequency_threshold_raises(self, bad_threshold):
        exp_matrix, gene_names = self._args()
        with pytest.raises(ValueError, match="frequency_threshold"):
            multi_run_flashscenic(
                exp_matrix, gene_names, frequency_threshold=bad_threshold,
            )

    def test_seeds_length_mismatch_raises(self):
        exp_matrix, gene_names = self._args()
        with pytest.raises(ValueError, match="seeds length"):
            multi_run_flashscenic(
                exp_matrix, gene_names, n_runs=3, seeds=[1, 2],
            )

    def test_n_runs_below_one_raises(self):
        exp_matrix, gene_names = self._args()
        with pytest.raises(ValueError, match="n_runs must be"):
            multi_run_flashscenic(exp_matrix, gene_names, n_runs=0)


class _FakeTrainer:
    def __init__(self, exp_matrix, n_steps, device):
        self._n_genes = exp_matrix.shape[1]

    def train(self):
        pass

    def get_adj(self):
        return np.zeros((self._n_genes, self._n_genes), dtype=np.float16)


class _StopEarly(Exception):
    pass


def test_cv_threshold_with_frequency_warns(monkeypatch):
    """cv_threshold only affects aggregation_method='mean_adjacency'; setting
    it together with aggregation_method='frequency' should warn since it
    would otherwise silently have no effect."""
    import regdiffusion

    monkeypatch.setattr(regdiffusion, "RegDiffusionTrainer", _FakeTrainer)
    monkeypatch.setattr(fs_data, "download_data", lambda **kwargs: (_ for _ in ()).throw(_StopEarly()))

    exp_matrix = np.zeros((3, 5), dtype=np.float32)
    gene_names = [f"g{i}" for i in range(5)]

    with pytest.warns(UserWarning, match="cv_threshold has no effect"):
        with pytest.raises(_StopEarly):
            multi_run_flashscenic(
                exp_matrix, gene_names,
                n_runs=1, aggregation_method="frequency", cv_threshold=0.5,
                verbose=False,
            )


def test_no_warning_for_cv_threshold_with_mean_adjacency(monkeypatch):
    import regdiffusion

    monkeypatch.setattr(regdiffusion, "RegDiffusionTrainer", _FakeTrainer)
    monkeypatch.setattr(fs_data, "download_data", lambda **kwargs: (_ for _ in ()).throw(_StopEarly()))

    exp_matrix = np.zeros((3, 5), dtype=np.float32)
    gene_names = [f"g{i}" for i in range(5)]

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        with pytest.raises(_StopEarly):
            multi_run_flashscenic(
                exp_matrix, gene_names,
                n_runs=1, aggregation_method="mean_adjacency", cv_threshold=0.5,
                verbose=False,
            )
    assert not any("cv_threshold has no effect" in str(w.message) for w in record)
