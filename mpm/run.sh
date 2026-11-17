#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
exec conda run --no-capture-output -n MPM_couple python -B "$ROOT/src/tools/u4_joint.py" "$@"
