from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from staged_attribution.export_tokens import export_token_records
from staged_attribution.token_protocol import (
    TokenProtocolError,
    parse_answer,
    serialize_answer,
    special_tokens,
)


def test_exact_htp_serialization_and_round_trip() -> None:
    codes = torch.tensor([[0, 1, 2, 3], [31, 63, 127, 255]])

    answer = serialize_answer([1, 0, 1], codes)

    assert answer == (
        "<t_5>, <|t_begin|><|p_begin|><a_0><b_1><c_2><d_3><|p_end|>, "
        "<|p_begin|><a_31><b_63><c_127><d_255><|p_end|><|t_end|>"
    )
    parsed = parse_answer(answer)
    assert parsed["lens_type"] == [1, 0, 1]
    assert parsed["codes"] == [[0, 1, 2, 3], [31, 63, 127, 255]]


@pytest.mark.parametrize(
    ("lens_type", "codes", "message"),
    [
        ([1, 0], [[0, 0, 0, 0]], "three bits"),
        ([1, 2, 0], [[0, 0, 0, 0]], "binary"),
        ([1, 0, 0], [], "at least one event"),
        ([1, 0, 0], [[32, 0, 0, 0]], "layer 0"),
        ([1, 0, 0], [[0, 64, 0, 0]], "layer 1"),
        ([1, 0, 0], [[0, 0, 128, 0]], "layer 2"),
        ([1, 0, 0], [[0, 0, 0, 256]], "layer 3"),
    ],
)
def test_rejects_invalid_lens_or_codes(lens_type, codes, message: str) -> None:
    with pytest.raises(TokenProtocolError, match=message):
        serialize_answer(lens_type, codes)


def test_special_tokens_preserve_baseline_order() -> None:
    tokens = special_tokens(num_roads=2)

    assert tokens[:12] == [
        "<|t_begin|>", "<|t_end|>", "<|p_begin|>", "<|p_end|>",
        "<t_0>", "<t_1>", "<t_2>", "<t_3>",
        "<t_4>", "<t_5>", "<t_6>", "<t_7>",
    ]
    assert tokens[12] == "<a_0>"
    assert tokens[43] == "<a_31>"
    assert tokens[44] == "<b_0>"
    assert tokens[-2:] == ["<road_0>", "<road_1>"]
    assert len(tokens) == 4 + 8 + 32 + 64 + 128 + 256 + 2


def test_export_is_sorted_hashed_and_population_complete(tmp_path: Path) -> None:
    records = [
        {"traj_id": 9, "lens_type": [0, 0, 1], "codes": [[1, 2, 3, 4]], "road_seq": [2]},
        {"traj_id": 3, "lens_type": [1, 1, 0], "codes": [[5, 6, 7, 8]], "road_seq": [1, 0]},
    ]
    output = tmp_path / "tokens.jsonl"

    audit = export_token_records(records, expected_traj_ids=[3, 9], output_path=output)

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["traj_id"] for row in rows] == [3, 9]
    assert rows[0]["answer"].startswith("<t_6>, <|t_begin|>")
    assert audit["population"] == 2
    assert len(audit["sha256"]) == 64
    assert audit["traj_id_sha256"] != audit["sha256"]


def test_export_rejects_duplicate_missing_and_length_mismatch(tmp_path: Path) -> None:
    base = {"traj_id": 3, "lens_type": [0, 0, 0], "codes": [[1, 2, 3, 4]], "road_seq": [1]}
    with pytest.raises(TokenProtocolError, match="duplicate"):
        export_token_records([base, dict(base)], [3], tmp_path / "duplicate.jsonl")
    with pytest.raises(TokenProtocolError, match="population"):
        export_token_records([base], [3, 9], tmp_path / "missing.jsonl")
    bad = dict(base, codes=[[1, 2, 3, 4], [2, 3, 4, 5]], code_length=1)
    with pytest.raises(TokenProtocolError, match="code_length"):
        export_token_records([bad], [3], tmp_path / "length.jsonl")
