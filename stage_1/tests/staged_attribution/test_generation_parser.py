from __future__ import annotations

import pytest

from staged_attribution.generate_stage3 import (
    GenerationError,
    format_generation_prompt,
    parse_generation,
    validate_generation_population,
)
from staged_attribution.token_protocol import serialize_answer


def valid_answer() -> str:
    return serialize_answer([1, 0, 1], [[1, 2, 3, 4], [5, 6, 7, 8]])


def test_valid_generation_is_parsed_without_prompt_roads() -> None:
    continuation = valid_answer() + "<|im_end|>"

    record = parse_generation(17, continuation)

    assert record["status"] == "completed"
    assert record["traj_id"] == 17
    assert record["lens_type"] == [1, 0, 1]
    assert record["codes"] == [[1, 2, 3, 4], [5, 6, 7, 8]]
    assert record["raw_continuation"] == continuation


@pytest.mark.parametrize(
    ("continuation", "reason"),
    [
        ("nothing useful", "missing_sequence"),
        ("<t_0>, <|t_begin|><|p_begin|><a_1><b_2><c_3><d_4><|p_end|>", "truncated"),
        (valid_answer() + " and again " + valid_answer(), "duplicate_sequence"),
        ("<t_0>, <|t_begin|><|p_begin|><a_32><b_2><c_3><d_4><|p_end|><|t_end|>", "invalid_protocol"),
    ],
)
def test_failures_are_explicit_records(continuation: str, reason: str) -> None:
    record = parse_generation(9, continuation)

    assert record["status"] == "failed"
    assert record["failure"]["reason"] == reason
    assert record["raw_continuation"] == continuation


def test_population_validation_rejects_drop_or_duplicate() -> None:
    records = [parse_generation(1, valid_answer()), parse_generation(2, "bad")]

    assert validate_generation_population(records, [1, 2]) == {"population": 2, "completed": 1, "failed": 1}
    with pytest.raises(GenerationError, match="population"):
        validate_generation_population(records[:1], [1, 2])
    with pytest.raises(GenerationError, match="duplicate"):
        validate_generation_population([records[0], records[0]], [1])


def test_generation_prompt_uses_oracle_route_description_only() -> None:
    prompt = format_generation_prompt("description with <road_3>")

    assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert "description with <road_3>" in prompt
