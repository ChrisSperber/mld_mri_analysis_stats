"""Extract tract-wise MRI measures for the documented analysis sample.

Median values of DTI, DKI, NODDI and FOD-derived scalar maps are extracted
within subject-space fibre tract masks. Previously extracted fixel-wise
median FD, FC and FDC values are added from the fixel metrics table.

The sample reference table determines the included subjects and sessions.

Outputs:
- CSV table with one row per subject, session and tract.
- TXT report summarising feature extraction completeness.
"""

# %%

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from mld_mri_stats.config import (
    DATA_MASTER_DIR,
    FIXEL_METRICS_CSV,
    NOT_AVAILABLE,
)
from mld_mri_stats.feature_extraction import extract_roi_quantile
from mld_mri_stats.utils import (
    Cols,
    FibreTractMasks,
    PathCols,
    load_nifti,
)

SCRIPT_PATH = Path(__file__)
REFERENCE_PATH = SCRIPT_PATH.with_name("a_document_sample_reference.csv")
OUTPUT_PATH = SCRIPT_PATH.with_suffix(".csv")
REPORT_PATH = SCRIPT_PATH.with_name(f"{SCRIPT_PATH.stem}_report.txt")

KEY_COLS = [Cols.SUBJECT_ID, Cols.SESSION_ID, "Tract"]

MRI_METRICS = {
    "FA": PathCols.FA,
    "MD": PathCols.MD,
    "AK": PathCols.DKI_AK,
    "MK": PathCols.DKI_MK,
    "RK": PathCols.DKI_RK,
    "NDI": PathCols.NODDI_NDI,
    "ODI": PathCols.NODDI_ODI,
    "FWF": PathCols.NODDI_FWF,
    "FOD_integral": PathCols.FOD_INTEGRAL,
}

TRACTS = {
    FibreTractMasks.CC1: PathCols.TRACT_CC1,
    FibreTractMasks.CC2: PathCols.TRACT_CC2,
    FibreTractMasks.CC3: PathCols.TRACT_CC3,
    FibreTractMasks.CC4: PathCols.TRACT_CC4,
    FibreTractMasks.CC5: PathCols.TRACT_CC5,
    FibreTractMasks.CST_L: PathCols.TRACT_CST_L,
    FibreTractMasks.CST_R: PathCols.TRACT_CST_R,
    FibreTractMasks.IFOF_L: PathCols.TRACT_IFOF_L,
    FibreTractMasks.IFOF_R: PathCols.TRACT_IFOF_R,
    FibreTractMasks.ILF_L: PathCols.TRACT_ILF_L,
    FibreTractMasks.ILF_R: PathCols.TRACT_ILF_R,
    FibreTractMasks.SLF1_L: PathCols.TRACT_SLF1_L,
    FibreTractMasks.SLF1_R: PathCols.TRACT_SLF1_R,
    FibreTractMasks.SLF2_L: PathCols.TRACT_SLF2_L,
    FibreTractMasks.SLF2_R: PathCols.TRACT_SLF2_R,
    FibreTractMasks.SLF3_L: PathCols.TRACT_SLF3_L,
    FibreTractMasks.SLF3_R: PathCols.TRACT_SLF3_R,
}

FIXEL_METRICS = {
    "FD_median": "FD",
    "FC_median": "FC",
    "FDC_median": "FDC",
}

# %%
# Load sample reference

df = pd.read_csv(
    REFERENCE_PATH,
    sep=";",
    na_values=[NOT_AVAILABLE],
    dtype={
        Cols.SUBJECT_ID: str,
        Cols.SESSION_ID: str,
    },
)


# Check availability of all required images and tract masks
required_paths = list(MRI_METRICS.values()) + list(TRACTS.values())

missing_paths = df[required_paths].isna()

if missing_paths.any().any():
    missing = df.loc[
        missing_paths.any(axis=1),
        [Cols.SUBJECT_ID, Cols.SESSION_ID] + required_paths,
    ]

    raise ValueError(f"Missing required image paths:\n{missing.to_string(index=False)}")

# %%
# Extract voxel-wise MRI metrics

results = []

for _, session in tqdm(
    df.iterrows(),
    total=len(df),
    desc="Extracting tract measures",
):
    subject_id = session[Cols.SUBJECT_ID]
    session_id = session[Cols.SESSION_ID]

    # Load all scalar MRI maps once per session
    images = {}
    affines = {}

    for metric, column in MRI_METRICS.items():
        image = load_nifti(
            DATA_MASTER_DIR / session[column],
            canonical=True,
        )

        images[metric] = image.get_fdata()
        affines[metric] = image.affine

    # Iterate over tract masks
    for tract, column in TRACTS.items():
        mask_image = load_nifti(
            DATA_MASTER_DIR / session[column],
            canonical=True,
        )

        mask = mask_image.get_fdata()

        result = {
            Cols.SUBJECT_ID: subject_id,
            Cols.SESSION_ID: session_id,
            "Tract": tract,
        }

        # Extract median of each MRI metric within the current tract
        for metric, image in images.items():
            context = f"{subject_id}, {session_id}, {tract}, {metric}"

            if image.shape != mask.shape:
                raise ValueError(
                    f"Shape mismatch: {context}. "
                    f"Image: {image.shape}, mask: {mask.shape}."
                )

            if not np.allclose(
                affines[metric],
                mask_image.affine,  # type: ignore
                atol=1e-4,
                rtol=0,
            ):
                raise ValueError(f"Affine mismatch: {context}.")

            try:
                result[metric] = extract_roi_quantile(
                    image,
                    mask,
                    quantile=0.5,
                )
            except ValueError as error:
                raise ValueError(f"{context}: {error}") from error

        results.append(result)

features = pd.DataFrame(results)


# %%
# Load and integrate previously extracted fixel metrics

fixel_metrics = pd.read_csv(
    FIXEL_METRICS_CSV,
    sep=",",
    na_values=[NOT_AVAILABLE],
    dtype={
        Cols.SUBJECT_ID: str,
        Cols.SESSION_ID: str,
    },
)

fixel_metrics = fixel_metrics[KEY_COLS + list(FIXEL_METRICS)].rename(
    columns=FIXEL_METRICS
)

features = features.merge(
    fixel_metrics,
    on=KEY_COLS,
    how="left",
    validate="one_to_one",
)

# Check availability of all fixel metrics
fixel_columns = list(FIXEL_METRICS.values())

for column in fixel_columns:
    features[column] = pd.to_numeric(features[column], errors="raise")

missing_fixels = features[fixel_columns].isna().any(axis=1)

if missing_fixels.any():
    missing = features.loc[missing_fixels, KEY_COLS + fixel_columns]

    raise ValueError(f"Missing fixel metrics:\n{missing.to_string(index=False)}")


# %%
# Save extracted features

features.to_csv(
    OUTPUT_PATH,
    sep=";",
    index=False,
    na_rep=NOT_AVAILABLE,
)


# %%
# Generate extraction report

metric_columns = list(MRI_METRICS) + fixel_columns

summary = pd.DataFrame(
    {
        "Available": features[metric_columns].notna().sum(),
        "Missing": features[metric_columns].isna().sum(),
    }
)

missing_by_tract = (
    features.groupby("Tract")[metric_columns]
    .agg(lambda values: values.isna().sum())
    .reindex(list(TRACTS))
)

with REPORT_PATH.open("w", encoding="utf-8") as file:
    file.write("TRACT-WISE MRI FEATURE EXTRACTION\n")
    file.write("=================================\n\n")

    file.write(f"Sample reference: {REFERENCE_PATH.name}\n")
    file.write(f"Fixel metrics source: {FIXEL_METRICS_CSV}\n\n")

    file.write(f"N subjects: {df[Cols.SUBJECT_ID].nunique()}\n")
    file.write(f"N sessions: {len(df)}\n")
    file.write(f"N tracts: {len(TRACTS)}\n")
    file.write(f"N MRI metrics: {len(metric_columns)}\n")
    file.write(f"Expected rows: {len(df) * len(TRACTS)}\n")
    file.write(f"Extracted rows: {len(features)}\n")
    file.write("Summary statistic: median\n\n")

    file.write("FEATURE COMPLETENESS\n")
    file.write("--------------------\n")
    file.write(summary.to_string())
    file.write("\n\n")

    file.write("MISSING VALUES BY TRACT\n")
    file.write("-----------------------\n")
    file.write(missing_by_tract.to_string())
    file.write("\n")

print(f"Tract measures saved to: {OUTPUT_PATH}")
print(f"Extraction report saved to: {REPORT_PATH}")

# %%
