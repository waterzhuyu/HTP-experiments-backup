from __future__ import annotations

import re
from contextlib import contextmanager
from contextvars import ContextVar
from collections.abc import Sequence

import numpy as np
import torch


VOCAB_SIZES = (32, 64, 128, 256)
_ACTIVE_SIZES = ContextVar("rq_vocab_sizes", default=VOCAB_SIZES)

def current_vocab_sizes():
    return _ACTIVE_SIZES.get()

@contextmanager
def codebook_protocol(sizes):
    sizes = tuple(sizes)
    if len(sizes) != 4 or any(type(k) is not int or k < 2 for k in sizes):
        raise TokenProtocolError("exactly four integer codebook sizes >= 2 required")
    token = _ACTIVE_SIZES.set(sizes)
    try:
        yield sizes
    finally:
        _ACTIVE_SIZES.reset(token)

LENS_BITS = 3
PREFIXES = ("a", "b", "c", "d")
BOUNDARY_TOKENS = ("<|t_begin|>", "<|t_end|>", "<|p_begin|>", "<|p_end|>")


class TokenProtocolError(ValueError):
    """Raised when a Semantic ID violates the frozen HTP wire protocol."""


def _lens_bits(values: Sequence[int] | torch.Tensor) -> tuple[int, int, int]:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().tolist()
    bits = tuple(int(value) for value in values)
    if len(bits) != LENS_BITS:
        raise TokenProtocolError("lens_type must contain exactly three bits")
    if any(value not in (0, 1) for value in bits):
        raise TokenProtocolError("lens_type bits must be binary")
    return bits


def _code_rows(values: object) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        array = values.detach().cpu().numpy()
    else:
        array = np.asarray(values)
    if array.size == 0:
        raise TokenProtocolError("codes must contain at least one event")
    if array.ndim != 2 or array.shape[1] != len(current_vocab_sizes()):
        raise TokenProtocolError("codes must have shape [events, 4]")
    if not np.issubdtype(array.dtype, np.integer):
        if not np.isfinite(array).all() or not np.equal(array, np.floor(array)).all():
            raise TokenProtocolError("codes must contain integers")
    array = array.astype(np.int64, copy=False)
    for layer, size in enumerate(current_vocab_sizes()):
        if int(array[:, layer].min()) < 0 or int(array[:, layer].max()) >= size:
            raise TokenProtocolError(f"codes in layer {layer} must be inside [0, {size})")
    return array


def lens_type_token(lens_type: Sequence[int] | torch.Tensor) -> str:
    bits = _lens_bits(lens_type)
    value = int("".join(str(bit) for bit in bits), 2)
    return f"<t_{value}>"


def event_tokens(codes: object) -> list[str]:
    rows = _code_rows(codes)
    return [
        "".join(f"<{PREFIXES[layer]}_{int(value)}>" for layer, value in enumerate(row))
        for row in rows
    ]


def serialize_answer(lens_type: Sequence[int] | torch.Tensor, codes: object) -> str:
    events = [f"<|p_begin|>{value}<|p_end|>" for value in event_tokens(codes)]
    sequence = "<|t_begin|>" + ", ".join(events) + "<|t_end|>"
    return f"{lens_type_token(lens_type)}, {sequence}"


_ANSWER = re.compile(r"^<t_([0-7])>, <\|t_begin\|>(.+)<\|t_end\|>$")
_EVENT = re.compile(
    r"^<\|p_begin\|><a_(\d+)><b_(\d+)><c_(\d+)><d_(\d+)><\|p_end\|>$"
)


def parse_answer(answer: str) -> dict[str, object]:
    if not isinstance(answer, str):
        raise TokenProtocolError("answer must be a string")
    match = _ANSWER.fullmatch(answer)
    if match is None:
        raise TokenProtocolError("answer does not match the HTP sequence envelope")
    value = int(match.group(1))
    bits = [int(bit) for bit in format(value, "03b")]
    raw_events = match.group(2).split(", ")
    codes: list[list[int]] = []
    for index, raw_event in enumerate(raw_events):
        event = _EVENT.fullmatch(raw_event)
        if event is None:
            raise TokenProtocolError(f"malformed pattern event {index}")
        codes.append([int(code) for code in event.groups()])
    validated = _code_rows(codes)
    return {"lens_type": bits, "codes": validated.tolist()}


def special_tokens(num_roads: int) -> list[str]:
    if not isinstance(num_roads, int) or num_roads < 0:
        raise TokenProtocolError("num_roads must be a non-negative integer")
    result = list(BOUNDARY_TOKENS)
    result.extend(f"<t_{value}>" for value in range(2**LENS_BITS))
    for prefix, size in zip(PREFIXES, current_vocab_sizes()):
        result.extend(f"<{prefix}_{value}>" for value in range(size))
    result.extend(f"<road_{value}>" for value in range(num_roads))
    return result
