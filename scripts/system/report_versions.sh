#!/usr/bin/env bash
set -euo pipefail

# Report installed software versions.
# Run from anywhere; output is saved in reports/.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT_DIR="$REPO_DIR/reports"
mkdir -p "$OUT_DIR"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$OUT_DIR/system_versions_${STAMP}.txt"

{
    echo "=== System and Tool Versions ==="
    echo "Date (UTC): $(date -u)"
    echo

    echo "--- System ---"
    lsb_release -ds
    uname -r
    echo

    echo "--- FreeSurfer / SynthSeg ---"
    recon-all -version
    echo "SynthSeg: $(command -v mri_synthseg)"
    echo

    echo "--- FSL ---"
    cat "$FSLDIR/etc/fslversion"
    echo

    echo "--- ANTs ---"
    antsRegistration --version | head -n 1
    echo

    echo "--- Python ---"
    python --version
    echo "Executable: $(command -v python)"
    echo "Environment: ${VIRTUAL_ENV:-${CONDA_DEFAULT_ENV:-None}}"
    echo

    echo "--- TractSeg ---"
    TRACTSEG_PYTHON="${TRACTSEG_PYTHON:-python}"
    "$TRACTSEG_PYTHON" -m pip show TractSeg | grep -E '^(Name|Version):'
    echo

    echo "--- GPU ---"
    nvidia-smi --query-gpu=name,driver_version \
        --format=csv,noheader
    echo

    echo "--- Git ---"
    git -C "$REPO_DIR" rev-parse HEAD

} > "$LOG"

echo "Version report saved to: $LOG"
