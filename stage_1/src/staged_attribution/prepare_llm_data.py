from __future__ import annotations

import ast
import hashlib
import json
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


class LLMDataError(ValueError):
    """Raised when Stage2 data would violate population or prompt invariants."""


WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
DAYS = ("midnight", "morning", "afternoon", "evening")


def _road_name(value: object) -> str | None:
    if not isinstance(value, str) or value in {"Unnamed road", "未知道路", ""}:
        return None
    if value.startswith("["):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError) as error:
            raise LLMDataError(f"invalid road_name list literal: {value!r}") from error
        if not isinstance(parsed, (list, tuple)) or not parsed or not isinstance(parsed[0], str):
            return None
        value = parsed[0]
    return value


def _prompt(
    road_seq: list[int],
    road_names: list[object],
    road_lengths: list[float],
    time_seq: list[float],
    city: str,
) -> str:
    if not time_seq:
        raise LLMDataError("time sequence must be non-empty")
    if any(road < 0 or road >= len(road_names) for road in road_seq):
        raise LLMDataError("road_seq contains an out-of-range road ID")
    names = [_road_name(road_names[road]) for road in road_seq]
    names = [name for name in names if name is not None]
    names = [name for index, name in enumerate(names) if index == 0 or name != names[index - 1]]
    name_text = ", ".join(names)
    road_text = "".join(f"<road_{road}>" for road in road_seq)
    start = datetime.fromtimestamp(float(time_seq[0]), tz=timezone.utc)
    end = datetime.fromtimestamp(float(time_seq[-1]), tz=timezone.utc)
    seconds_total = int((end - start).total_seconds())
    minutes, seconds = divmod(seconds_total, 60)
    start_text = f"{WEEKDAYS[start.weekday()]} {DAYS[start.hour // 6]} at {start.hour:02d}:{start.minute:02d}"
    travel_time = f"{minutes} minutes and {seconds} seconds"
    travel_length = round(sum(float(road_lengths[road]) for road in road_seq) / 1000, 2)
    interval = 10 if city == "chengdu" else 15
    return (
        "Here is a summary of the user's travel information: "
        f"The road sequence is {road_text}, mainly passing through {name_text}. "
        f"The trip started at {start_text}, with an estimated travel time of {travel_time}, "
        f"and a total travel distance of approximately {travel_length} kilometers. "
        f"The GPS sampling interval is about one point every {interval} seconds. "
        "Please generate a sampling type and a travel pattern sequence of this trip."
    )


def build_llm_frame(
    token_records: Iterable[Mapping[str, object]],
    road_frame: pd.DataFrame,
    raw_frame: pd.DataFrame,
    city: str,
) -> pd.DataFrame:
    for column in ("road_name", "length"):
        if column not in road_frame:
            raise LLMDataError(f"road frame missing {column}")
    for column in ("traj_id", "user_id", "time", "cpath"):
        if column not in raw_frame:
            raise LLMDataError(f"raw frame missing {column}")
    if raw_frame["traj_id"].duplicated().any():
        raise LLMDataError("duplicate raw traj_id")
    tokens = [dict(record) for record in token_records]
    token_ids = [int(record["traj_id"]) for record in tokens]
    if len(set(token_ids)) != len(token_ids):
        raise LLMDataError("duplicate token traj_id")
    raw_by_id = raw_frame.set_index("traj_id", drop=False)
    raw_ids = {int(value) for value in raw_frame["traj_id"]}
    if set(token_ids) != raw_ids:
        raise LLMDataError(
            f"population mismatch: token_only={sorted(set(token_ids)-raw_ids)[:10]}, "
            f"raw_only={sorted(raw_ids-set(token_ids))[:10]}"
        )
    road_names = road_frame["road_name"].tolist()
    road_lengths = [float(value) for value in road_frame["length"]]
    rows: list[dict[str, object]] = []
    for token in tokens:
        traj_id = int(token["traj_id"])
        raw = raw_by_id.loc[traj_id]
        road_seq = [int(value) for value in token["road_seq"]]
        raw_road = [int(value) for value in raw["cpath"]]
        if road_seq != raw_road:
            raise LLMDataError(f"traj_id={traj_id} road_seq does not match raw cpath")
        rows.append(
            {
                "traj_id": traj_id,
                "desc": _prompt(
                    road_seq, road_names, road_lengths, [float(value) for value in raw["time"]], city
                ),
                "user_id": int(raw["user_id"]),
                "answer": str(token["answer"]),
            }
        )
    return pd.DataFrame(rows).sort_values("traj_id").reset_index(drop=True)


def deterministic_subset(traj_ids: Iterable[int], size: int, seed: int) -> list[int]:
    values = [int(value) for value in traj_ids]
    if len(set(values)) != len(values):
        raise LLMDataError("subset population contains duplicate traj_id")
    if not isinstance(size, int) or size <= 0 or size > len(values):
        raise LLMDataError("subset size must be positive and no larger than the population")
    ranked = sorted(
        values,
        key=lambda value: (hashlib.sha256(f"{seed}:{value}".encode()).digest(), value),
    )[:size]
    return sorted(ranked)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_llm_subsets(
    frame: pd.DataFrame,
    output_dir: str | Path,
    split: str,
    seed: int,
    sizes: Mapping[str, int | str],
) -> dict[str, Path]:
    if "traj_id" not in frame or frame["traj_id"].duplicated().any():
        raise LLMDataError("LLM frame requires unique traj_id")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    population_ids = sorted(int(value) for value in frame["traj_id"])
    frame_by_id = frame.set_index("traj_id", drop=False)
    result: dict[str, Path] = {}
    manifest: dict[str, object] = {
        "split": split,
        "seed": int(seed),
        "population": len(population_ids),
        "population_traj_id_sha256": hashlib.sha256(
            "\n".join(map(str, population_ids)).encode()
        ).hexdigest(),
        "subsets": {},
    }
    for name, requested in sizes.items():
        if requested == "full":
            selected = population_ids
        elif isinstance(requested, int):
            selected = deterministic_subset(population_ids, requested, seed)
        else:
            raise LLMDataError(f"invalid subset size for {name}: {requested!r}")
        subset = frame_by_id.loc[selected].reset_index(drop=True)
        path = output / f"{split}_{name}.parquet"
        temporary = path.with_suffix(".parquet.tmp")
        subset.to_parquet(temporary, index=False)
        temporary.replace(path)
        result[name] = path
        manifest["subsets"][name] = {
            "size": len(selected),
            "traj_ids": selected,
            "path": str(path.resolve()),
            "sha256": _file_hash(path),
        }
    manifest_path = output / f"{split}_subset_manifest.json"
    _atomic_json(manifest_path, manifest)
    result["manifest"] = manifest_path
    return result
