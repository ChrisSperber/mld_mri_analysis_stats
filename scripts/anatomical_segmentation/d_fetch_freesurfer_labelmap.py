"""Fetch and clean the FreeSurfer label map.

Combine labels used in SynthSeg segmentations with the FreeSurfer lookup table
and manually defined anatomical structure labels.

Output:
- CSV containing region IDs, names, hemispheres, and larger structures
"""

# %%

import os
from pathlib import Path

import numpy as np
import pandas as pd

from mld_mri_stats.config import REPO_ROOT, TEMPORARY_DATA_DIR
from mld_mri_stats.utils import load_nifti

SYNTHSEG_DIR = TEMPORARY_DATA_DIR / "synthseg"
SUFFIX_LABEL_MAP = "_synthseg_labels.nii.gz"

STRUCTURE_LABEL_MAP = (
    REPO_ROOT / "data" / "reference" / "freesurfer_labelmap_w_structure_labels.xlsx"
)

HEMISPHERE = "Hemisphere"
LABEL = "Label"

LEFT = "Left"
RIGHT = "Right"
NONE = "None"
RIGHT_TAGS = ["Right", "-rh-"]
LEFT_TAGS = ["Left", "-lh-"]

# %%
# Get FreeSurfer lookup table

freesurfer_home = Path(os.environ["FREESURFER_HOME"])
freesurfer_label_lut = freesurfer_home / "FreeSurferColorLUT.txt"

print(f"FreeSurfer Home Dir: {freesurfer_home}")

# %%
# Parse FreeSurfer LUT

records = []

with open(freesurfer_label_lut) as f:
    for txt_line in f:
        line = txt_line.strip()

        if not line or line.startswith("#"):
            continue

        parts = line.split()

        if len(parts) < 6:  # noqa: PLR2004
            continue

        try:
            idx = int(parts[0])
            name = " ".join(parts[1:-4])
        except ValueError:
            continue

        records.append({"id": idx, "name": name})

lut_df = pd.DataFrame.from_records(records).sort_values("id").reset_index(drop=True)

# %%
# Collect all labels present in SynthSeg segmentations

unique_values_set = set()

for file in SYNTHSEG_DIR.iterdir():
    if file.name.endswith(SUFFIX_LABEL_MAP):
        nifti = load_nifti(file)
        data = np.asarray(nifti.dataobj)
        unique_values_set.update(np.unique(data))

unique_values_list = sorted(int(value) for value in unique_values_set)

# %%
# Keep only labels present in the segmentations

lut_df_filtered = lut_df[lut_df["id"].isin(unique_values_list)].copy()
lut_df_filtered = lut_df_filtered.rename(columns={"name": LABEL})

# %%
# Add anatomical structure and hemisphere labels

structure_mapping_df = pd.read_excel(STRUCTURE_LABEL_MAP)

lut_df_filtered = lut_df_filtered.merge(
    right=structure_mapping_df,
    how="left",
    left_on=LABEL,
    right_on=LABEL,
)

lut_df_filtered[HEMISPHERE] = NONE

for index, row in lut_df_filtered.iterrows():
    label = row[LABEL]

    if any(tag in label for tag in RIGHT_TAGS):
        lut_df_filtered.loc[index, HEMISPHERE] = RIGHT
    elif any(tag in label for tag in LEFT_TAGS):
        lut_df_filtered.loc[index, HEMISPHERE] = LEFT

# %%
# Save lookup table

output_path = Path(__file__).with_suffix(".csv")
lut_df_filtered.to_csv(output_path, index=False, sep=";")

print(f"Label map saved to: {output_path}")

# %%
