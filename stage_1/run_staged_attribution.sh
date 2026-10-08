#!/usr/bin/env bash
set -euo pipefail

budget="${1:-smoke}"
shift || true

uv run python -m staged_attribution.run_experiment \
  --config stage_1/configs/staged_attribution/experiment.json \
  --budget "$budget" \
  "$@"
