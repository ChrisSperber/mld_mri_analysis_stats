#!/usr/bin/env bash
# Rigid registration of skull-stripped T1 images to native FA space.
# Store only ANTs affine transformations.

set -euo pipefail

# --------- Config ---------

PROJECT_DIR="/home/chris/projects"
DATA_MASTER_DIR="$PROJECT_DIR/mld_data_master_2026"
MASTER_CSV="$PROJECT_DIR/mld_mri_analysis_stats/data/reference/data_table_master.csv"

T1_DIR="$PROJECT_DIR/temp_images/synthseg"
OUTPUT_DIR="$PROJECT_DIR/temp_images/registration_T1_to_FA"

mkdir -p "$OUTPUT_DIR"

# set number of cores for ANTs
num_threads=$(nproc)
if (( num_threads > 1 )); then
    num_threads=$((num_threads - 1))
fi
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="$num_threads"


# --------- Read FA paths from master CSV ---------

read_fa_paths() {
    python3 - "$MASTER_CSV" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="", encoding="utf-8-sig") as f:
    for row in csv.DictReader(f, delimiter=";"):
        print("\t".join([
            row["SubjectID"],
            row["SessionID"],
            row["PathFA"],
        ]))
PY
}


# --------- Registration ---------

count=0

while IFS=$'\t' read -r subject session fa_relative; do

    base="sub-${subject}_ses-${session}"

    FA_PATH="$DATA_MASTER_DIR/$fa_relative"
    T1_PATH="$T1_DIR/${base}_brain.nii.gz"
    OUT_PREFIX="$OUTPUT_DIR/${base}_T1_to_FA_"

    # Skip already completed registrations
    if [[ -f "${OUT_PREFIX}0GenericAffine.mat" ]]; then
        echo "[$base] Registration already exists"
        continue
    fi

    echo "[$base] Rigid registration T1 -> FA"

    antsRegistration \
        --dimensionality 3 \
        --float 0 \
        --output "$OUT_PREFIX" \
        --interpolation Linear \
        --use-histogram-matching 0 \
        --winsorize-image-intensities "[0.01,0.99]" \
        --initial-moving-transform "[$FA_PATH,$T1_PATH,1]" \
        --transform "Rigid[0.1]" \
        --metric "MI[$FA_PATH,$T1_PATH,1,32,Regular,0.25]" \
        --convergence "[1000x500x250x100,1e-6,10]" \
        --shrink-factors 8x4x2x1 \
        --smoothing-sigmas 3x2x1x0vox

    count=$((count + 1))

done < <(read_fa_paths)

echo "Done. Completed $count registrations."
