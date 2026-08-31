from __future__ import annotations

import hashlib
import json
import os
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch


COMPACT_ADAPTER_FORMAT = "peft_reconstructible_base_embeddings_v1"
COMPACT_MANIFEST = "compact_adapter_manifest.json"
ADAPTER_MODEL = "adapter_model.safetensors"
RECONSTRUCTIBLE_SUFFIX = "token_adapter.base_layer.weight"


class AdapterStorageError(RuntimeError):
    """Raised when a compact PEFT adapter cannot be proven self-consistent."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _versions() -> dict[str, str]:
    return {
        "peft": version("peft"),
        "safetensors": version("safetensors"),
        "torch": str(torch.__version__),
        "transformers": version("transformers"),
    }


def _tensor_digest(tensor_slice: Any, rows: int, chunk_rows: int = 4096) -> str:
    digest = hashlib.sha256()
    for start in range(0, rows, chunk_rows):
        block = tensor_slice[start : start + chunk_rows].detach().cpu().contiguous()
        digest.update(block.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _safetensors_dtype(dtype: torch.dtype) -> str:
    labels = {
        torch.bfloat16: "BF16",
        torch.float16: "F16",
        torch.float32: "F32",
        torch.float64: "F64",
        torch.int8: "I8",
        torch.int16: "I16",
        torch.int32: "I32",
        torch.int64: "I64",
        torch.uint8: "U8",
        torch.bool: "BOOL",
    }
    try:
        return labels[dtype]
    except KeyError as error:
        raise AdapterStorageError(f"unsupported tensor dtype: {dtype}") from error


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AdapterStorageError(f"invalid {label}: {error}") from error
    if not isinstance(payload, dict):
        raise AdapterStorageError(f"{label} must contain a JSON object")
    return payload


def compact_peft_adapter(
    adapter_path: str | Path,
    *,
    base_model_path: str | Path,
    reconstruction_seed: int,
) -> dict[str, Any]:
    """Remove only PEFT base embeddings that the frozen load recipe rebuilds exactly.

    PEFT stores the resized input and output base embeddings alongside the LoRA and
    trainable-token deltas.  For this experiment those two tensors are byte-identical,
    deterministic functions of the immutable base model, token protocol, package
    versions, and training seed.  The deltas remain in every per-run adapter.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    root = Path(adapter_path).resolve()
    model_file = root / ADAPTER_MODEL
    config_file = root / "adapter_config.json"
    if not model_file.is_file() or not config_file.is_file():
        raise AdapterStorageError("adapter model and adapter_config.json must exist")
    adapter_config = _read_json(config_file, "adapter config")
    configured_base = Path(str(adapter_config.get("base_model_name_or_path", ""))).resolve()
    expected_base = Path(base_model_path).resolve()
    if configured_base != expected_base:
        raise AdapterStorageError(
            f"adapter base model mismatch: configured={configured_base}, expected={expected_base}"
        )
    base_config = expected_base / "config.json"
    if not base_config.is_file():
        raise AdapterStorageError(f"base model config is missing: {base_config}")

    original_bytes = model_file.stat().st_size
    original_sha256 = sha256_file(model_file)
    temporary_model = model_file.with_suffix(model_file.suffix + ".compact.tmp")
    manifest_file = root / COMPACT_MANIFEST
    temporary_manifest = root / (COMPACT_MANIFEST + ".tmp")
    stale = [path for path in (temporary_model, temporary_manifest, manifest_file) if path.exists()]
    if stale:
        raise AdapterStorageError(f"refusing compaction with existing state: {stale}")

    with safe_open(model_file, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        removed = sorted(key for key in keys if key.endswith(RECONSTRUCTIBLE_SUFFIX))
        if len(removed) != 2 or not any("embed_tokens" in key for key in removed) or not any(
            "lm_head" in key for key in removed
        ):
            raise AdapterStorageError(
                "expected exactly the input and output token_adapter base embeddings"
            )
        left = handle.get_slice(removed[0])
        right = handle.get_slice(removed[1])
        left_shape = tuple(int(value) for value in left.get_shape())
        right_shape = tuple(int(value) for value in right.get_shape())
        left_dtype = left.get_dtype()
        if left_shape != right_shape or left_dtype != right.get_dtype():
            raise AdapterStorageError("reconstructible base embedding shapes or dtypes differ")
        for start in range(0, left_shape[0], 4096):
            if not torch.equal(left[start : start + 4096], right[start : start + 4096]):
                raise AdapterStorageError(
                    f"input/output base embeddings differ at row chunk {start}"
                )
        shared_digest = _tensor_digest(left, left_shape[0])
        tensors = {key: handle.get_tensor(key) for key in keys if key not in removed}
        metadata = handle.metadata()

    save_file(tensors, temporary_model, metadata=metadata)
    with safe_open(temporary_model, framework="pt", device="cpu") as handle:
        compact_keys = list(handle.keys())
    if any(key in compact_keys for key in removed) or len(compact_keys) != len(keys) - len(removed):
        temporary_model.unlink(missing_ok=True)
        raise AdapterStorageError("compact adapter tensor inventory is invalid")

    compact_bytes = temporary_model.stat().st_size
    compact_sha256 = sha256_file(temporary_model)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "format": COMPACT_ADAPTER_FORMAT,
        "adapter_model": ADAPTER_MODEL,
        "adapter_model_sha256_before_compaction": original_sha256,
        "adapter_model_sha256": compact_sha256,
        "adapter_config_sha256": sha256_file(config_file),
        "base_model_name_or_path": str(expected_base),
        "base_model_config_sha256": sha256_file(base_config),
        "reconstruction_seed": int(reconstruction_seed),
        "package_versions": _versions(),
        "removed_tensors": [
            {
                "name": key,
                "shape": list(left_shape),
                "dtype": left_dtype,
                "sha256": shared_digest,
            }
            for key in removed
        ],
        "original_bytes": original_bytes,
        "compact_bytes": compact_bytes,
        "saved_bytes": original_bytes - compact_bytes,
        "reconstruction_recipe": (
            "set_seed(reconstruction_seed); load immutable base model; "
            "resize_token_embeddings(tokenizer_length, mean_resizing=True); load PEFT deltas"
        ),
    }
    with temporary_manifest.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    # Publish the manifest first.  A crash before the model replacement then fails
    # closed on SHA validation instead of silently loading a compact model as legacy.
    temporary_manifest.replace(manifest_file)
    temporary_model.replace(model_file)
    return payload


def validate_compact_adapter(adapter_path: str | Path) -> dict[str, Any] | None:
    root = Path(adapter_path).resolve()
    manifest_path = root / COMPACT_MANIFEST
    if not manifest_path.is_file():
        return None
    payload = _read_json(manifest_path, "compact adapter manifest")
    if payload.get("schema_version") != 1 or payload.get("format") != COMPACT_ADAPTER_FORMAT:
        raise AdapterStorageError("unsupported compact adapter manifest")
    model_file = root / str(payload.get("adapter_model", ""))
    config_file = root / "adapter_config.json"
    if not model_file.is_file() or not config_file.is_file():
        raise AdapterStorageError("compact adapter files are missing")
    if sha256_file(model_file) != payload.get("adapter_model_sha256"):
        raise AdapterStorageError("compact adapter model SHA256 mismatch")
    if sha256_file(config_file) != payload.get("adapter_config_sha256"):
        raise AdapterStorageError("compact adapter config SHA256 mismatch")
    expected_versions = payload.get("package_versions")
    if expected_versions != _versions():
        raise AdapterStorageError(
            f"compact adapter package version mismatch: {expected_versions} != {_versions()}"
        )
    base_model = Path(str(payload.get("base_model_name_or_path", ""))).resolve()
    base_config = base_model / "config.json"
    if not base_config.is_file() or sha256_file(base_config) != payload.get(
        "base_model_config_sha256"
    ):
        raise AdapterStorageError("compact adapter base model config mismatch")
    adapter_config = _read_json(config_file, "adapter config")
    if Path(str(adapter_config.get("base_model_name_or_path", ""))).resolve() != base_model:
        raise AdapterStorageError("compact adapter base model path mismatch")
    try:
        int(payload["reconstruction_seed"])
    except (KeyError, TypeError, ValueError) as error:
        raise AdapterStorageError("compact adapter reconstruction seed is invalid") from error
    return payload


def reconstruction_seed_for_adapter(adapter_path: str | Path, fallback: int) -> int:
    payload = validate_compact_adapter(adapter_path)
    return int(payload["reconstruction_seed"]) if payload is not None else int(fallback)


def validate_loaded_compact_adapter(model: Any, adapter_path: str | Path) -> None:
    """Prove that the loader reconstructed every omitted base embedding exactly."""
    payload = validate_compact_adapter(adapter_path)
    if payload is None:
        return
    expected = {
        item["name"]: item
        for item in payload.get("removed_tensors", [])
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    if len(expected) != 2:
        raise AdapterStorageError("compact adapter removed tensor inventory is invalid")
    try:
        named_parameters = model.named_parameters(remove_duplicate=False)
    except TypeError:  # pragma: no cover - compatibility with older wrappers
        named_parameters = model.named_parameters()
    actual = {
        name: parameter
        for name, parameter in named_parameters
        if name in expected
    }
    if set(actual) != set(expected):
        raise AdapterStorageError(
            f"loaded compact adapter tensor inventory mismatch: {sorted(actual)}"
        )
    for name, parameter in actual.items():
        item = expected[name]
        shape = [int(value) for value in parameter.shape]
        if shape != item.get("shape") or _safetensors_dtype(parameter.dtype) != item.get("dtype"):
            raise AdapterStorageError(f"loaded compact adapter tensor metadata mismatch: {name}")
        if _tensor_digest(parameter, shape[0]) != item.get("sha256"):
            raise AdapterStorageError(f"loaded compact adapter tensor SHA256 mismatch: {name}")
