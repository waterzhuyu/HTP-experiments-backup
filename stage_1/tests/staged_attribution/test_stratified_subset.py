from __future__ import annotations

import pytest

from staged_attribution.stratified_subset import (
    REGISTERED_STRATA,
    StratifiedSubsetError,
    hamilton_quotas,
    lens_type_bits,
    stratified_subset,
    verify_token_lens_mapping,
)


def test_lens_type_bits_match_three_htp_encoder_length_parities() -> None:
    assert lens_type_bits(20) == (0, 0, 1)
    assert lens_type_bits(21) == (1, 1, 0)
    assert lens_type_bits(24) == (0, 0, 0)
    with pytest.raises(StratifiedSubsetError, match="positive"):
        lens_type_bits(0)


def test_hamilton_quotas_are_exact_and_lexically_tie_broken() -> None:
    quotas = hamilton_quotas({"000": 3, "001": 3, "010": 2}, size=5)
    assert quotas == {"000": 2, "001": 2, "010": 1}
    assert sum(quotas.values()) == 5


def test_stratified_subset_is_order_independent_and_preserves_all_strata() -> None:
    ids = list(range(80))
    lengths = [24 + (index % 8) for index in ids]
    first = stratified_subset(ids, lengths, size=16, seed=42)
    second = stratified_subset(reversed(ids), reversed(lengths), size=16, seed=42)

    assert first == second
    assert set(first["strata"]) == set(REGISTERED_STRATA)
    assert all(row["quota"] == 2 for row in first["strata"].values())
    assert len(first["selected_traj_ids"]) == 16
    assert first != stratified_subset(ids, lengths, size=16, seed=43)


def test_stratified_subset_handles_rare_strata_without_overallocation() -> None:
    ids = list(range(12))
    lengths = [24] * 5 + [25] * 4 + [26] * 2 + [27]
    result = stratified_subset(
        ids, lengths, size=9, seed=7, require_all_registered_strata=False
    )

    assert sum(row["quota"] for row in result["strata"].values()) == 9
    assert all(row["quota"] <= row["population"] for row in result["strata"].values())


def test_stratified_subset_rejects_population_errors() -> None:
    with pytest.raises(StratifiedSubsetError, match="align"):
        stratified_subset([1, 2], [24], size=1, seed=1)
    with pytest.raises(StratifiedSubsetError, match="duplicate"):
        stratified_subset([1, 1], [24, 25], size=1, seed=1)
    with pytest.raises(StratifiedSubsetError, match="all eight"):
        stratified_subset([1, 2], [24, 25], size=1, seed=1)


def test_verify_token_lens_mapping_requires_exact_population_and_formula() -> None:
    records = [
        {"traj_id": 2, "lens_type": [0, 0, 1]},
        {"traj_id": 7, "lens_type": [1, 1, 0]},
    ]
    result = verify_token_lens_mapping(records, {2: 20, 7: 21})
    assert result["population"] == 2
    assert result["formula_match"] is True

    with pytest.raises(StratifiedSubsetError, match="mismatch"):
        verify_token_lens_mapping(records, {2: 20, 7: 22})
    with pytest.raises(StratifiedSubsetError, match="populations differ"):
        verify_token_lens_mapping(records[:1], {2: 20, 7: 21})
