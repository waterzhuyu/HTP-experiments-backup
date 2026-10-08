from __future__ import annotations

import numpy as np
import torch

from multiview_pattern.metrics import (
    bootstrap_metrics,
    contingency,
    distribution_similarity,
    grouped_permutation,
    metrics_from_contingency,
    optimal_mapping,
    reindex_matrix,
)
from multiview_pattern.model import VIEWS, IndependentTokenizer


def batch() -> dict[str, torch.Tensor]:
    return {
        "traj_id": torch.tensor([1, 2]),
        "gps": torch.rand(2, 17, 2),
        "grid": torch.randint(0, 20, (2, 17)),
        "road": torch.randint(0, 11, (2, 17)),
        "lengths": torch.tensor([17, 13]),
    }


def test_scope_excludes_poi() -> None:
    assert VIEWS == ("gps", "grid", "road")


def test_both_encoder_families_produce_one_code_per_ceil_eight_points() -> None:
    for family in ("transformer", "gru"):
        model = IndependentTokenizer(
            view="gps",
            encoder_family=family,
            num_grids=20,
            road_features=torch.randn(11, 8),
            hidden_dim=16,
            latent_dim=8,
            codebook_size=8,
            dropout=0.0,
        )
        output = model(batch(), 5, 4)
        assert torch.isfinite(output.loss)
        assert output.mask.sum(dim=1).tolist() == [3, 2]
        assert torch.all(output.codes[~output.mask] == -1)


def test_each_tokenizer_has_private_codebook_and_encoder() -> None:
    kwargs = {
        "encoder_family": "transformer",
        "num_grids": 20,
        "road_features": torch.randn(11, 8),
        "hidden_dim": 16,
        "latent_dim": 8,
        "codebook_size": 8,
        "dropout": 0.0,
    }
    gps = IndependentTokenizer(view="gps", **kwargs)
    grid = IndependentTokenizer(view="grid", **kwargs)
    assert gps.quantizer.embedding.weight.data_ptr() != grid.quantizer.embedding.weight.data_ptr()
    assert gps.encoder.output.weight.data_ptr() != grid.encoder.output.weight.data_ptr()


def test_bfloat16_autocast_preserves_scatter_dtype() -> None:
    model = IndependentTokenizer(
        view="road",
        encoder_family="gru",
        num_grids=20,
        road_features=torch.randn(11, 8),
        hidden_dim=16,
        latent_dim=8,
        codebook_size=8,
        dropout=0.0,
    )
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = model(batch(), 5, 4)
    assert torch.isfinite(output.loss)


def test_permutation_invariant_metrics_and_mapping() -> None:
    left = np.tile(np.arange(4), 100)
    right = np.asarray([2, 0, 3, 1], dtype=np.int64)[left]
    matrix = contingency(left, right, 4)
    metrics = metrics_from_contingency(matrix)
    assert metrics["adjusted_mutual_information"] > 0.999
    assert metrics["optimal_match_rate"] == 1.0
    assert metrics["normalized_h_right_given_left"] < 1e-9
    mapping = optimal_mapping(right, left, 4)
    np.testing.assert_array_equal(mapping[right], left)


def test_directional_conditional_entropy_detects_many_to_one() -> None:
    left = np.tile(np.arange(4), 100)
    right = left // 2
    metrics = metrics_from_contingency(contingency(left, right, 4))
    assert metrics["h_right_given_left"] < 1e-9
    assert metrics["h_left_given_right"] > 0.6


def test_grouped_shuffle_preserves_groups_and_avoids_same_trajectory() -> None:
    labels = np.repeat(np.arange(4), 8)
    trajectories = np.tile(np.arange(8), 4)
    permutation, metadata = grouped_permutation(labels, trajectories, 7)
    np.testing.assert_array_equal(labels[permutation], labels)
    assert metadata["same_trajectory_fraction"] == 0.0


def test_aligned_block_similarity_is_one() -> None:
    matrix = np.arange(16, dtype=np.float64).reshape(4, 4)
    left_mapping = np.asarray([2, 0, 3, 1])
    right_mapping = np.asarray([1, 3, 0, 2])
    permuted = np.zeros_like(matrix)
    permuted[np.ix_(np.argsort(left_mapping), np.argsort(right_mapping))] = matrix
    aligned = reindex_matrix(permuted, left_mapping, right_mapping)
    similarity = distribution_similarity(matrix, aligned)
    assert similarity["cosine_similarity"] > 0.999999
    assert similarity["jensen_shannon_divergence"] < 1e-12


def test_trajectory_bootstrap_reports_fast_metrics() -> None:
    left = np.asarray([0, 0, 1, 1, 2, 2, 3, 3])
    right = np.asarray([1, 1, 2, 2, 3, 3, 0, 0])
    offsets = np.asarray([0, 2, 4, 6, 8])
    result = bootstrap_metrics(left, right, offsets, size=4, seed=9, replicates=5)
    assert set(result) == {
        "normalized_mutual_information",
        "optimal_match_rate",
        "normalized_h_right_given_left",
        "normalized_h_left_given_right",
    }
    assert result["optimal_match_rate"]["lower_95"] > 0.99
