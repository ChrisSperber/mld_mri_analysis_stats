"""Apply geodesic white matter subparcellation to the CH2 template.

Uses the existing SynthSeg segmentation and FreeSurfer anatomical labels.
The output retains the original CH2 image geometry and FreeSurfer label IDs.

Output:
- wm subparcellation map via geodesic voronoi algorithm ch2_wm_subparcellation.nii.gz
    in temp_images/ch2_pipeline_demo
"""

# %%
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

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

IMAGE_DIR = PROJECTS_DIR / "temp_images" / "ch2_pipeline_demo"
INPUT_FILE = IMAGE_DIR / "ch2_synthseg_labels.nii.gz"
OUTPUT_FILE = IMAGE_DIR / "ch2_wm_subparcellation.nii.gz"
LABELMAP_FILE = SCRIPT_DIR / "d_fetch_freesurfer_labelmap.csv"

WM_LABELS = {"Left": 2, "Right": 41}

# %%
# Load segmentation and anatomical label definitions
img = load_nifti(INPUT_FILE, canonical=False)
data = np.asarray(img.dataobj).astype(np.int32)
spacing = img.header.get_zooms()[:3]

labelmap = pd.read_csv(LABELMAP_FILE, sep=";")
labelmap = labelmap[
    ~labelmap["Structure"].isin(NON_REQUIRED_LABEL_STRUCTURES_WM_SUBPARCELLATION)
]

# %%
# Subparcellate each hemisphere
results = {}

for hemisphere in ("Left", "Right"):
    wm_mask = data == WM_LABELS[hemisphere]

    seed_ids = labelmap.loc[labelmap["Hemisphere"] == hemisphere, "id"].to_numpy()

    seeds = np.where(np.isin(data, seed_ids), data, 0)

    results[hemisphere] = geodesic_subparcellate(
        wm_mask=wm_mask,
        seed_labels=seeds,
        spacing=spacing,
    )

# %%
# Merge hemispheres and save output
result = merge_hemisphere_labelmaps(results["Left"], results["Right"])

output_img = nib.Nifti1Image(
    result,
    affine=img.affine,
    header=img.header.copy(),
)
output_img.set_data_dtype(np.int32)
nib.save(output_img, OUTPUT_FILE)

print(f"Saved: {OUTPUT_FILE}")

# %%
