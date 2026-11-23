"""Generate geodesic white matter subparcellations from SynthSeg label maps.

For each SynthSeg segmentation, cerebral white matter voxels are assigned
the anatomical label of the nearest cortical or subcortical seed region,
using shortest paths constrained to white matter and adjacent seed voxels.

Left and right hemispheres are processed separately. The output preserves
the original FreeSurfer label IDs and image geometry.

Inputs
- synthseg T1 segmentation maps in projects/temp_images/synthseg/*_synthseg_labels.nii.gz

Outputs
- subparcellation maps of the WM masks in projects/temp_images/white_matter_subparcellation/
    *_wm_subparcellation.nii.gz

"""

# %%
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm

from mld_mri_stats.utils import (
    NON_REQUIRED_LABEL_STRUCTURES_WM_SUBPARCELLATION,
    geodesic_subparcellate,
    load_nifti,
    merge_hemisphere_labelmaps,
)

# %%
# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECTS_DIR = SCRIPT_DIR.parents[2]

INPUT_DIR = PROJECTS_DIR / "temp_images" / "synthseg"
OUTPUT_DIR = PROJECTS_DIR / "temp_images" / "white_matter_subparcellation"
LABELMAP_FILE = SCRIPT_DIR / "d_fetch_freesurfer_labelmap.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# FreeSurfer cerebral white matter labels
WM_LABELS = {"Left": 2, "Right": 41}

# %%
# Load FreeSurfer label definitions
labelmap = pd.read_csv(LABELMAP_FILE, sep=";")

labelmap = labelmap[
    ~labelmap["Structure"].isin(NON_REQUIRED_LABEL_STRUCTURES_WM_SUBPARCELLATION)
]

# Select anatomical seed labels separately for each hemisphere
seed_ids = {
    hemisphere: labelmap.loc[labelmap["Hemisphere"] == hemisphere, "id"].to_numpy()
    for hemisphere in ("Left", "Right")
}

# %%
# Process SynthSeg segmentations
files = sorted(INPUT_DIR.glob("*_synthseg_labels.nii.gz"))

for input_file in tqdm(files, desc="WM subparcellation"):
    output_file = OUTPUT_DIR / input_file.name.replace(
        "_synthseg_labels.nii.gz",
        "_wm_subparcellation.nii.gz",
    )

    if output_file.exists():
        continue

    img = load_nifti(input_file, canonical=False)
    data = np.asarray(img.dataobj).astype(np.int32)
    spacing = img.header.get_zooms()[:3]

    hemisphere_results = {}

    for hemisphere in ("Left", "Right"):
        wm_mask = data == WM_LABELS[hemisphere]

        seeds = np.where(
            np.isin(data, seed_ids[hemisphere]),
            data,
            0,
        )

        hemisphere_results[hemisphere] = geodesic_subparcellate(
            wm_mask=wm_mask,
            seed_labels=seeds,
            spacing=spacing,
        )

    result = merge_hemisphere_labelmaps(
        hemisphere_results["Left"],
        hemisphere_results["Right"],
    )

    output_img = nib.Nifti1Image(
        result,
        affine=img.affine,
        header=img.header.copy(),
    )
    output_img.set_data_dtype(np.int32)
    nib.save(output_img, output_file)

print(f"Finished WM subparcellation. Outputs: {OUTPUT_DIR}")

# %%
