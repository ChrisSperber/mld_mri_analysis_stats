#!/usr/bin/env bash
# Download the CH2 template and segment it using the current T1 pipeline.

set -euo pipefail

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
OUTDIR="$(dirname "$PROJECT_DIR")/temp_images/ch2_pipeline_demo"
mkdir -p "$OUTDIR"

# Files
CH2="$OUTDIR/ch2.nii.gz"
CROPPED="$OUTDIR/ch2_cropped.nii.gz"
SEGMENTATION="$OUTDIR/ch2_synthseg_labels.nii.gz"

URL="https://raw.githubusercontent.com/neurolabusc/MRIcron/master/Resources/templates/ch2.nii.gz"

# 1. Download CH2 template
if [[ ! -f "$CH2" ]]; then
    echo "Downloading CH2 template..."
    curl -fL "$URL" -o "$CH2"
fi

# 2. Crop field of view (same procedure as T1 pipeline)
if [[ ! -f "$CROPPED" ]]; then
    echo "Cropping CH2..."
    robustfov -i "$CH2" -r "$CROPPED"
fi

# 3. SynthSeg segmentation
if [[ ! -f "$SEGMENTATION" ]]; then
    echo "Running SynthSeg..."
    mri_synthseg \
        --i "$CROPPED" \
        --o "$SEGMENTATION" \
        --threads 1 \
        --keepgeom \
        --parc \
        --robust
fi

echo "Finished."
echo "Template: $CROPPED"
echo "Segmentation: $SEGMENTATION"
