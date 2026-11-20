"""Generate fibre orientation peak maps from normalized WM FODs.

Use the DWI brain masks to restrict peak extraction.

Requirements:
- Code in /scripts/data_documentation_and_validation was run
- the mrtrix3 conda enviroment was set up; this is accsessed through subprocess

Output:
- Peak maps in temp_images/fod_peaks/
"""

# %%

import subprocess

import pandas as pd

from mld_mri_stats.config import (
    DATA_DIFFUSION_DERIVATIVES_DIR,
    DATA_MASTER_DIR,
    MASTER_DATA_CSV,
    NOT_AVAILABLE,
    TEMPORARY_DATA_DIR,
)
from mld_mri_stats.utils import Cols, PathCols

OUTPUT_DIR = TEMPORARY_DATA_DIR / "fod_peaks"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %%
# Read master table

main_data_df = pd.read_csv(
    MASTER_DATA_CSV,
    sep=";",
    dtype={Cols.SUBJECT_ID: str, Cols.SESSION_ID: str},
)

# %%
# Generate FOD peaks

for _, row in main_data_df.iterrows():
    subject = row[Cols.SUBJECT_ID]
    session = row[Cols.SESSION_ID]
    relative_path = row[PathCols.FOD_WM_NORM]

    if pd.isna(relative_path) or relative_path == NOT_AVAILABLE:
        continue

    base = f"sub-{subject}_ses-{session}"

    fod_path = DATA_MASTER_DIR / relative_path
    mask_path = (
        DATA_DIFFUSION_DERIVATIVES_DIR
        / f"sub-{subject}"
        / f"ses-{session}"
        / "dwi"
        / f"{base}_desc-brain_mask.mif"
    )
    output_path = OUTPUT_DIR / f"{base}_peaks.nii.gz"

    if output_path.exists():
        print(f"Skipping existing: {base}")
        continue

    print(f"Generating FOD peaks: {base}")

    subprocess.run(  # noqa: S603
        [  # noqa: S607
            "conda",
            "run",
            "-n",
            "mrtrix3",
            "sh2peaks",
            str(fod_path),
            str(output_path),
            "-mask",
            str(mask_path),
            "-num",
            "3",
        ],
        check=True,
    )

print("FOD peak generation completed.")

# %%
