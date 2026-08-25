from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from staged_attribution.prepare_llm_data import (
    LLMDataError,
    build_llm_frame,
    deterministic_subset,
    write_llm_subsets,
)


def token_records() -> list[dict[str, object]]:
    return [
        {
            "traj_id": 7,
            "road_seq": [0, 1, 1],
            "answer": "<t_0>, <|t_begin|><|p_begin|><a_0><b_0><c_0><d_0><|p_end|><|t_end|>",
        },
        {
            "traj_id": 2,
            "road_seq": [1],
            "answer": "<t_1>, <|t_begin|><|p_begin|><a_1><b_1><c_1><d_1><|p_end|><|t_end|>",
        },
    ]


def raw_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "traj_id": [2, 7],
            "user_id": [20, 70],
            "time": [[0, 75], [86400 + 3600, 86400 + 3725]],
            "cpath": [[1], [0, 1, 1]],
        }
    )


def road_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {"road_name": ["Alpha", "Beta"], "length": [1000.0, 500.0]}
    )


def test_prompt_and_answer_match_original_htp_contract() -> None:
    frame = build_llm_frame(token_records(), road_frame(), raw_frame(), city="porto")

    assert frame["traj_id"].tolist() == [2, 7]
    first = frame.iloc[0]
    assert first["answer"] == token_records()[1]["answer"]
    assert first["user_id"] == 20
    assert first["desc"] == (
        "Here is a summary of the user's travel information: "
        "The road sequence is <road_1>, mainly passing through Beta. "
        "The trip started at Thursday midnight at 00:00, with an estimated travel time of 1 minutes and 15 seconds, "
        "and a total travel distance of approximately 0.5 kilometers. "
        "The GPS sampling interval is about one point every 15 seconds. "
        "Please generate a sampling type and a travel pattern sequence of this trip."
    )


@pytest.mark.parametrize(
    ("tokens", "raw", "message"),
    [
        (token_records()[:1], raw_frame(), "population"),
        (token_records(), pd.concat([raw_frame(), raw_frame().iloc[[0]]]), "duplicate raw"),
        ([dict(token_records()[0], road_seq=[1]), token_records()[1]], raw_frame(), "road_seq"),
    ],
)
def test_rejects_population_duplicate_and_route_mismatch(tokens, raw, message: str) -> None:
    with pytest.raises(LLMDataError, match=message):
        build_llm_frame(tokens, road_frame(), raw, city="porto")


def test_deterministic_subset_is_order_independent() -> None:
    ids = list(range(100))
    selected = deterministic_subset(ids, size=12, seed=42)

    assert selected == deterministic_subset(reversed(ids), size=12, seed=42)
    assert selected != deterministic_subset(ids, size=12, seed=43)
    assert len(selected) == 12
    assert len(set(selected)) == 12


def test_write_subsets_records_explicit_manifest(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {"traj_id": list(range(20)), "desc": ["d"] * 20, "user_id": list(range(20)), "answer": ["a"] * 20}
    )

    result = write_llm_subsets(
        frame,
        output_dir=tmp_path,
        split="test",
        seed=314159,
        sizes={"smoke": 4, "controlled": 8, "full": "full"},
    )

    assert set(result) == {"smoke", "controlled", "full", "manifest"}
    manifest = json.loads((tmp_path / "test_subset_manifest.json").read_text())
    assert manifest["population"] == 20
    assert len(manifest["subsets"]["smoke"]["traj_ids"]) == 4
    assert len(manifest["subsets"]["controlled"]["traj_ids"]) == 8
    assert manifest["subsets"]["full"]["traj_ids"] == list(range(20))
    assert all(Path(path).is_file() for name, path in result.items() if name != "manifest")
