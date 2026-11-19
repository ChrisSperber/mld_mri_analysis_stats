#!/usr/bin/env bash
set -euo pipefail

# Segment T1 images with SynthSeg and generate brain masks.
# Inputs are read from the local analysis master CSV.
# All outputs are stored in temp_images/synthseg.
# steps: crop -> skull stripping & cortical segmentation
# NOTE: Freesurfer >v8.x, FSL, and ANTs are required. ANTs can be installed according to
# https://github.com/ANTsX/ANTs/wiki/Compiling-ANTs-on-Linux-and-Mac-OS (requiring also cmake and a c++ compiler) and adding
# it to the PATH
# Freesurfer 8.x currently (10/2025) does NOT run on Ubuntu 24
#
# Requirements: Code in /scripts/data_documentation_and_validation was run

# --------- Config ---------

PROJECT_DIR="/home/chris/projects"
DATA_MASTER_DIR="$PROJECT_DIR/mld_data_master_2026"
MASTER_CSV="$PROJECT_DIR/mld_mri_analysis_stats/data/reference/data_table_master.csv"
OUT_DIR="$PROJECT_DIR/temp_images/synthseg"

# set n threads; keep low to prevent RAM allocation errors
threads=1
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="$threads"
export LC_NUMERIC=C

mkdir -p "$OUT_DIR"

# --------- Tool checks ---------

for cmd in python3 mri_synthseg fslmaths robustfov; do
    command -v "$cmd" >/dev/null 2>&1 || {
        echo "Required tool not found: $cmd"
        exit 1
    }
done

timestamp() { date +"%Y-%m-%d %H:%M:%S"; }

# --------- Process one T1 image ---------

process_one() {
    local subject="$1"
    local session="$2"
    local relative_path="$3"

    local base="sub-${subject}_ses-${session}"
    local in_t1="$DATA_MASTER_DIR/$relative_path"

    local cropped="$OUT_DIR/${base}_T1_cropped.nii.gz"
    local seg="$OUT_DIR/${base}_synthseg_labels.nii.gz"
    local mask="$OUT_DIR/${base}_brain_mask.nii.gz"
    local brain="$OUT_DIR/${base}_brain.nii.gz"

    echo "[$(timestamp)] >>> Start: $base"

    if [[ ! -f "$in_t1" ]]; then
        echo "  ERROR: T1 not found: $in_t1"
        return 1
    fi

    # 1) Crop neck to reduce image size
    if [[ ! -f "$cropped" ]]; then
        echo "[$(timestamp)] - robustfov"
        robustfov -i "$in_t1" -r "$cropped"
    else
        echo "[$(timestamp)] - skip cropping (exists)"
    fi

    # 2) SynthSeg segmentation and cortical parcellation
    if [[ ! -f "$seg" ]]; then
        echo "[$(timestamp)] - SynthSeg"

        # --robust is memory intensive; retain original settings
        mri_synthseg \
            --i "$cropped" \
            --o "$seg" \
            --threads "$threads" \
            --keepgeom \
            --parc \
            --robust
    else
        echo "[$(timestamp)] - skip SynthSeg (exists)"
    fi

    # 3) Brain mask from nonzero segmentation labels
    if [[ ! -f "$mask" ]]; then
        echo "[$(timestamp)] - Brain mask"
        fslmaths "$seg" -thr 1 -bin "$mask"
    else
        echo "[$(timestamp)] - skip mask (exists)"
    fi

    # 4) Skull stripping
    if [[ ! -f "$brain" ]]; then
        echo "[$(timestamp)] - Skull stripping"
        fslmaths "$cropped" -mas "$mask" "$brain"
    else
        echo "[$(timestamp)] - skip skull stripping (exists)"
    fi

    echo "[$(timestamp)] >>> Finished: $base"
}

# --------- Read T1 paths from master CSV ---------

read_t1_paths() {
    python3 - "$MASTER_CSV" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="", encoding="utf-8-sig") as f:
    for row in csv.DictReader(f, delimiter=";"):
        path = row["PathT1"]

        if not path or path == "Not_available":
            continue

        print("\t".join([
            row["SubjectID"],
            row["SessionID"],
            path,
        ]))
PY
}

# --------- Main loop ---------

count=0

while IFS=$'\t' read -r subject session relative_path; do
    process_one "$subject" "$session" "$relative_path"
    count=$((count + 1))
done < <(read_t1_paths)

echo "[$(timestamp)] Processed $count T1 images"
echo "Output directory: $OUT_DIR"
