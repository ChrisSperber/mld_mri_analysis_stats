#!/usr/bin/env bash
# Run TractSeg on FOD peak maps.
#
# Outputs per subject/session:
# - Postprocessed binary tract masks
# - Voxel-wise tract probability maps
#
# TractSeg preprocessing was disabled to prevent errors.
# NOTE: pytorch 2.5.1 was installed to match the currently tested pytorch version of tractseg 2.10

set -euo pipefail

# --------- Paths ---------

PROJECT_DIR="/home/chris/projects"
PEAK_DIR="$PROJECT_DIR/temp_images/fod_peaks"
OUT_DIR="$PROJECT_DIR/temp_images/tractseg"

mkdir -p "$OUT_DIR"

# --------- Process peak images ---------

for peak_path in "$PEAK_DIR"/*_peaks.nii.gz; do
    filename="$(basename "$peak_path")"
    base="${filename%_peaks.nii.gz}"
    subject_out="$OUT_DIR/$base"

    echo "Processing: $base"

    mkdir -p "$subject_out"

    # 1. Standard postprocessed binary segmentations
    if [[ ! -d "$subject_out/bundle_segmentations" ]]; then
        echo "  Generating binary segmentations"

        TractSeg \
            -i "$peak_path" \
            -o "$subject_out"
    else
        echo "  Binary segmentations already exist"
    fi

    # 2. Probability maps without postprocessing
    if [[ ! -d "$subject_out/bundle_probabilities" ]]; then
        echo "  Generating probability maps"

        TractSeg \
            -i "$peak_path" \
            -o "$subject_out" \
            --get_probabilities \
            --tract_segmentation_output_dir bundle_probabilities
    else
        echo "  Probability maps already exist"
    fi

    echo "  Done"
    echo
done

echo "TractSeg processing completed."
