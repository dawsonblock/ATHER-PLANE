#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV="${AWA_VENV:-$REPO_ROOT/.venv}"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found; use a CUDA-enabled RunPod image." >&2
  exit 2
fi
VOLUME_PATH="${AWA_RUNPOD_VOLUME_PATH:-/workspace}"
VOLUME_SIZE_GIB="${AWA_RUNPOD_VOLUME_SIZE_GB:-}"
VOLUME_ID="${AWA_RUNPOD_VOLUME_ID:-}"
if ! [[ "$VOLUME_SIZE_GIB" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "ERROR: set AWA_RUNPOD_VOLUME_SIZE_GB from the provisioned RunPod network-volume quota." >&2
  echo "P0 does not infer quota from the container's backing filesystem." >&2
  exit 3
fi
if [[ -z "$VOLUME_ID" ]]; then
  echo "ERROR: set AWA_RUNPOD_VOLUME_ID to the attached RunPod network volume ID." >&2
  exit 3
fi
if [[ ! -d "$VOLUME_PATH" ]] || ! mountpoint -q "$VOLUME_PATH"; then
  echo "ERROR: persistent RunPod volume is not mounted at $VOLUME_PATH." >&2
  exit 3
fi
USED_BYTES="$(du -sx --block-size=1 "$VOLUME_PATH" | awk '{print $1}')"
if ! [[ "$USED_BYTES" =~ ^[0-9]+$ ]]; then
  echo "ERROR: cannot measure persistent-volume usage at $VOLUME_PATH." >&2
  exit 3
fi
USED_GIB="$(python3 -c 'import sys; print(int(sys.argv[1]) / (1024**3))' "$USED_BYTES")"
REMAINING_GIB="$(python3 -c 'import sys; print(float(sys.argv[1]) - float(sys.argv[2]))' "$VOLUME_SIZE_GIB" "$USED_GIB")"
if ! python3 -c 'import sys; raise SystemExit(0 if float(sys.argv[1]) >= 80.0 else 1)' "$REMAINING_GIB"; then
  echo "ERROR: v2.38.6 needs at least 80 GiB of provisioned persistent-volume quota remaining." >&2
  echo "Quota=${VOLUME_SIZE_GIB} GiB; measured usage=${USED_GIB} GiB." >&2
  exit 3
fi
nvidia-smi
"$PYTHON_BIN" -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev,doom-vjepa]"
mkdir -p runs/v2_38_6
awa-v2-split-check
awa-v2-progressive-executor --config configs/v2_38_progressive_executor.yaml --evidence-root runs/v2_38_6
cat <<'EOF'
Bootstrap complete.
Review the printed next-phase plan. To execute exactly one eligible phase:
  awa-v2-progressive-executor --config configs/v2_38_progressive_executor.yaml --evidence-root runs/v2_38_6 --execute-next
The five-seed DREAM-RSI phase requires the additional --allow-expensive flag.
EOF
