"""Independently audit all codebook-ablation tuple receipts against the primary ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import subprocess
from datetime import datetime, timezone


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def close(actual: float, expected: float, message: str) -> None:
    require(math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12), message)


def check_event_summary(summary: dict, trajectories: list[dict], key: str) -> None:
    events = sum(item[key]["events"] for item in trajectories)
    correct = sum(item[key]["tuple_correct"] for item in trajectories)
    layers = [sum(item[key]["layer_correct"][i] for item in trajectories) for i in range(4)]
    require(events == 5996 == summary["events"], f"{key}: event denominator")
    require(correct == summary["tuple_correct"], f"{key}: tuple numerator")
    require(layers == summary["layer_correct"], f"{key}: layer numerators")
    close(summary["tuple_accuracy"], correct / events, f"{key}: tuple accuracy")
    for i in range(4):
        close(summary["layer_accuracy"][i], layers[i] / events, f"{key}: layer {i} accuracy")
    require(sum(summary["first_wrong_layer_counts"].values()) == events, f"{key}: first-error partition")
    require(sum(summary["correctness_pattern_counts"].values()) == events, f"{key}: pattern partition")


def audit_one(root: pathlib.Path, analysis: pathlib.Path, ledger: dict, row: dict, validation_ids: set[int]) -> dict | None:
    setting, seed, cell = ledger["setting"], ledger["seed"], ledger["cell"]
    candidates = sorted(analysis.glob(f"tuple_{setting}_{seed}_{cell}_cpu_*/verification.json"))
    completed = [path for path in candidates if json.loads(path.read_text())["status"] == "completed"]
    if not completed:
        return None
    require(len(completed) == 1, f"{setting}/{seed}/{cell}: {len(completed)} completed receipts")
    receipt_path = completed[0]
    run_dir = receipt_path.parent
    receipt = json.loads(receipt_path.read_text())
    require(receipt["analysis_revision"] in {
        "32493fdfc2240ae141ba8abd668566bfdb1b07e3",
        "b5ef8e928da8aea1d3203958c83723bfa872f1e3",
    }, f"{setting}/{seed}/{cell}: unreviewed analysis revision")
    manifest = pathlib.Path(ledger["path"]) / "run_manifest.json"
    require(manifest.exists(), f"{setting}/{seed}/{cell}: missing source manifest")
    require(sha256(manifest) == ledger["manifest_sha256"] == receipt["source_manifest_sha256"], f"{setting}/{seed}/{cell}: source manifest SHA")
    require(receipt["training_revision"] == ledger["revision"], f"{setting}/{seed}/{cell}: training revision")
    require(receipt["population"] == 1024, f"{setting}/{seed}/{cell}: receipt population")
    require(receipt["evaluation_device"] == "cpu" and receipt["evaluation_dtype"] == "float32", f"{setting}/{seed}/{cell}: execution dtype/device")
    require(receipt["sid_nll_absolute_difference"] <= 0.005, f"{setting}/{seed}/{cell}: SID NLL parity")
    for relative, expected_sha in receipt["files"].items():
        path = run_dir / relative
        require(path.exists() and sha256(path) == expected_sha, f"{setting}/{seed}/{cell}: {relative} SHA")
    metrics = json.loads((run_dir / cell / "metrics.json").read_text())
    progress = json.loads((run_dir / cell / "progress.json").read_text())
    require(metrics["status"] == progress["status"] == "completed", f"{setting}/{seed}/{cell}: metrics/progress status")
    require(progress["population_completed"] == progress["population_total"] == 1024, f"{setting}/{seed}/{cell}: progress population")
    require(metrics["population"] == 1024 and metrics["seed"] == seed, f"{setting}/{seed}/{cell}: metrics population/seed")
    require(metrics["restoration_verified_before_dtype_conversion"] is True, f"{setting}/{seed}/{cell}: BF16 restoration")
    require(metrics["analysis_revision"] == receipt["analysis_revision"] and metrics["training_revision"] == receipt["training_revision"], f"{setting}/{seed}/{cell}: revisions")
    require(metrics["source_manifest_sha256"] == receipt["source_manifest_sha256"], f"{setting}/{seed}/{cell}: metrics source SHA")
    close(metrics["original_sid_nll"], row["sid_nll"], f"{setting}/{seed}/{cell}: primary SID NLL")
    require(metrics["evaluation_device"] == "cpu" and metrics["evaluation_dtype"] == "float32", f"{setting}/{seed}/{cell}: metrics dtype/device")
    require(metrics["vocab_sizes"] == row["rq_vocab_sizes"], f"{setting}/{seed}/{cell}: codebook capacities")
    require(metrics["source_artifact_count"] == ledger["artifacts_verified"], f"{setting}/{seed}/{cell}: source artifacts")
    trajectory = metrics["per_trajectory"]
    require(len(trajectory) == 1024 and {item["traj_id"] for item in trajectory} == validation_ids, f"{setting}/{seed}/{cell}: trajectory IDs")
    check_event_summary(metrics["teacher_forced"], trajectory, "teacher_forced")
    check_event_summary(metrics["free_running"], trajectory, "free_running")
    require(all(metrics["reevaluated_metrics"]["denominators"][f"layer_{i}"] == 5996 for i in range(4)), f"{setting}/{seed}/{cell}: NLL layer denominators")
    require(metrics["reevaluated_metrics"]["denominators"]["sampling"] == 1024, f"{setting}/{seed}/{cell}: sampling denominator")
    close(metrics["sid_nll_absolute_difference"], receipt["sid_nll_absolute_difference"], f"{setting}/{seed}/{cell}: NLL parity receipt")
    close(abs(metrics["reevaluated_metrics"]["sid_nll"] - metrics["original_sid_nll"]), receipt["sid_nll_absolute_difference"], f"{setting}/{seed}/{cell}: recomputed NLL parity")
    close(metrics["teacher_forced"]["tuple_accuracy"], receipt["teacher_tuple_accuracy"], f"{setting}/{seed}/{cell}: teacher receipt")
    close(metrics["free_running"]["tuple_accuracy"], receipt["free_tuple_accuracy"], f"{setting}/{seed}/{cell}: free receipt")
    return {
        "setting": setting, "seed": seed, "cell": cell,
        "receipt_path": str(receipt_path), "verification_sha256": sha256(receipt_path),
        "source_manifest_sha256": receipt["source_manifest_sha256"],
        "analysis_revision": receipt["analysis_revision"],
        "elapsed_seconds": receipt["elapsed_seconds"],
        "sid_nll_absolute_difference": receipt["sid_nll_absolute_difference"],
        "teacher_tuple_accuracy": receipt["teacher_tuple_accuracy"],
        "free_tuple_accuracy": receipt["free_tuple_accuracy"],
        "teacher_tuple_correct": metrics["teacher_forced"]["tuple_correct"],
        "free_tuple_correct": metrics["free_running"]["tuple_correct"],
        "free_running_failed_trajectories": metrics["free_running_failed_trajectories"],
        "free_running_extra_events": metrics["free_running_extra_events"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=pathlib.Path, required=True)
    parser.add_argument("--snapshot", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    snapshot = json.loads(args.snapshot.read_text())
    require(snapshot["verified_runs"] == 30 and len(snapshot["ledger"]) == 30, "primary snapshot does not cover 30 runs")
    validation_ids = set(snapshot["validation_ids"])
    require(len(validation_ids) == 1024, "primary validation ID set is not 1024")
    rows = {(row["setting"], row["seed"], row["cell"]): row for row in snapshot["rows"]}
    require(len(rows) == 30, "primary rows are not 30 unique runs")
    ledger_keys = [(entry["setting"], entry["seed"], entry["cell"]) for entry in snapshot["ledger"]]
    require(len(set(ledger_keys)) == 30 and set(ledger_keys) == set(rows), "primary ledger/rows do not cover the same 30 unique runs")
    require(all(entry["verified"] and len(entry["attempts"]) == 1 and entry["attempts"][0]["status"] == "completed" for entry in snapshot["ledger"]), "primary ledger contains an unverified or repeated training run")
    results, pending, issues = [], [], []
    for ledger in snapshot["ledger"]:
        key = (ledger["setting"], ledger["seed"], ledger["cell"])
        try:
            result = audit_one(args.root, args.root / "analysis", ledger, rows[key], validation_ids)
            (results if result else pending).append(result if result else {"setting": key[0], "seed": key[1], "cell": key[2]})
        except (KeyError, ValueError, OSError, TypeError) as exc:
            issues.append({"setting": key[0], "seed": key[1], "cell": key[2], "error": str(exc)})
    result_map = {(r["setting"], r["seed"], r["cell"]): r for r in results}
    pairs = []
    for setting in snapshot["capacity_order"]:
        for seed in snapshot["seed_order"]:
            b2, h1 = result_map.get((setting, seed, "B2")), result_map.get((setting, seed, "H1"))
            if b2 and h1:
                pairs.append({"setting": setting, "seed": seed,
                              "teacher_h1_minus_b2": h1["teacher_tuple_accuracy"] - b2["teacher_tuple_accuracy"],
                              "free_h1_minus_b2": h1["free_tuple_accuracy"] - b2["free_tuple_accuracy"]})
    status = "failed" if issues else "completed" if len(results) == 30 and len(pairs) == 15 else "partial"
    git_revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or None
    output = {"status": status, "compiled_at": datetime.now(timezone.utc).isoformat(),
              "analysis_script_revision": git_revision, "primary_snapshot_sha256": sha256(args.snapshot),
              "verified_runs": len(results), "expected_runs": 30, "paired_seed_comparisons": len(pairs),
              "max_sid_nll_absolute_difference": max((r["sid_nll_absolute_difference"] for r in results), default=None),
              "cpu_audit_elapsed_hours": sum(r["elapsed_seconds"] for r in results) / 3600,
              "teacher_h1_higher_pairs": sum(p["teacher_h1_minus_b2"] > 0 for p in pairs),
              "free_h1_higher_pairs": sum(p["free_h1_minus_b2"] > 0 for p in pairs),
              "pending": pending, "issues": issues, "runs": results, "pairs": pairs,
              "interpretation": "B2 and H1 have different tokenizer-produced SID targets and different tuple entropy; cross-model tuple accuracy does not identify a shared-target predictive gain or G11 improvement."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: output[k] for k in ("status", "verified_runs", "paired_seed_comparisons", "max_sid_nll_absolute_difference", "issues")}, ensure_ascii=False))
    return 0 if not issues and (args.allow_partial or status == "completed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
