"""Validate NIfTI images listed in the local master data table.

Check image readability, dimensions, numerical values, spatial alignment of
diffusion-derived images, and consistency of binary mask formats.

NOTE: The code identified various issues, including 4D qT1 images with duplicated 3D images, and
falsely assinged RGB qT1 images. All issues have been fixed in the data master before continuing.
CAVEAT: MTR images generally contain some NaN/Inf values

Output:
- TXT report next to this script with results for every subject/session.
"""

# %%

from pathlib import Path

import numpy as np
import pandas as pd

from mld_mri_stats.config import DATA_MASTER_DIR, MASTER_DATA_CSV, NOT_AVAILABLE
from mld_mri_stats.utils import Cols, PathCols, load_nifti

# Images expected to share the FA voxel grid
DWI_COLS = [
    PathCols.FA,
    PathCols.MD,
    PathCols.DKI_AK,
    PathCols.DKI_MK,
    PathCols.DKI_RK,
    PathCols.NODDI_FWF,
    PathCols.NODDI_NDI,
    PathCols.NODDI_ODI,
    PathCols.FOD_INTEGRAL,
]

# Manual and FOD-derived masks
MANUAL_MASK_COLS = [
    PathCols.ROI_CST_L,
    PathCols.ROI_CST_R,
    PathCols.ROI_PLIC_L,
    PathCols.ROI_PLIC_R,
    PathCols.ROI_FWM_L,
    PathCols.ROI_FWM_R,
]

TRACT_MASK_COLS = [
    PathCols.TRACT_CC1,
    PathCols.TRACT_CC2,
    PathCols.TRACT_CC3,
    PathCols.TRACT_CC4,
    PathCols.TRACT_CC5,
    PathCols.TRACT_CST_L,
    PathCols.TRACT_CST_R,
    PathCols.TRACT_IFOF_L,
    PathCols.TRACT_IFOF_R,
    PathCols.TRACT_ILF_L,
    PathCols.TRACT_ILF_R,
    PathCols.TRACT_SLF1_L,
    PathCols.TRACT_SLF1_R,
    PathCols.TRACT_SLF2_L,
    PathCols.TRACT_SLF2_R,
    PathCols.TRACT_SLF3_L,
    PathCols.TRACT_SLF3_R,
]

MASK_GROUPS = {
    "Manual ROIs": MANUAL_MASK_COLS,
    "FOD tract masks": TRACT_MASK_COLS,
}

IMAGE_COLS = [
    PathCols.T1,
    PathCols.QT1,
    PathCols.FLAIR,
    PathCols.MTR,
    *DWI_COLS,
    *MANUAL_MASK_COLS,
    *TRACT_MASK_COLS,
]


# %%


def check_image(  # noqa: C901, PLR0912
    path: Path, column: str
) -> tuple[list[str], tuple | None, tuple | None]:
    """Check one NIfTI and return issues, geometry, and mask values."""
    issues = []

    try:
        img = load_nifti(path)
        data = np.asarray(img.dataobj)
    except Exception as exc:
        return [f"ERROR: {column}: Cannot read image: {exc}"], None, None

    if data.ndim != 3:  # noqa: PLR2004
        return [f"ERROR: {column}: Expected 3D, got {data.ndim}D"], None, None

    # Check numerical validity
    valid = np.isfinite(data)

    if not valid.all():
        n_invalid = np.count_nonzero(~valid)
        pct_invalid = 100 * n_invalid / data.size

        if column == PathCols.MTR:
            issues.append(
                f"WARNING: {column}: Contains {n_invalid} NaN/Inf voxels "
                f"({pct_invalid:.2f}%)"
            )
        else:
            issues.append(f"ERROR: {column}: Contains NaN or Inf")

        return issues, None, None

    if np.all(data == 0):
        issues.append(f"WARNING: {column}: All voxels are zero")
    elif np.min(data) == np.max(data):
        issues.append(f"WARNING: {column}: Constant image")

    # Basic intensity checks
    if column == PathCols.FA:
        if np.min(data) < -0.001 or np.max(data) > 1.25:  # noqa: PLR2004
            issues.append(f"WARNING: {column}: FA outside expected range (0-1.25)")

    if column in [
        PathCols.NODDI_FWF,
        PathCols.NODDI_NDI,
        PathCols.NODDI_ODI,
    ]:
        if np.min(data) < -0.001 or np.max(data) > 1.001:  # noqa: PLR2004
            issues.append(f"WARNING: {column}: Values outside 0-1")

    # Geometry: dimensions and voxel-to-world transformation
    geometry = (img.shape, img.affine.copy(), img.header.get_zooms()[:3])  # type: ignore

    # Mask format: unique numerical values, rounded for floating-point tolerance
    mask_values = None
    if column in MANUAL_MASK_COLS + TRACT_MASK_COLS:
        mask_values = tuple(np.unique(np.round(data, decimals=4)))
        if len(mask_values) > 2:  # noqa: PLR2004
            issues.append(f"WARNING: {column}: More than two mask values")

    return issues, geometry, mask_values


def check_geometry(geometry: tuple, reference: tuple, column: str) -> list[str]:
    """Compare an image voxel grid against the FA reference."""
    shape, affine, resolution = geometry
    ref_shape, ref_affine, ref_resolution = reference

    issues = []

    if shape != ref_shape:
        issues.append(f"ERROR: {column}: Shape differs from FA")

    if not np.allclose(resolution, ref_resolution, atol=1e-4, rtol=0):
        issues.append(f"ERROR: {column}: Resolution differs from FA")

    if not np.allclose(affine, ref_affine, atol=1e-3, rtol=0):
        issues.append(f"ERROR: {column}: Affine differs from FA")

    return issues


# %%
# Read master table

main_data_df = pd.read_csv(
    MASTER_DATA_CSV,
    sep=";",
    dtype={Cols.SUBJECT_ID: str, Cols.SESSION_ID: str},
)

report_lines = ["NIfTI validation report", "=" * 50, ""]

mask_formats = {}
results = []


# %%
# Validate images for each subject/session

for _, row in main_data_df.iterrows():
    subject = row[Cols.SUBJECT_ID]
    session = row[Cols.SESSION_ID]
    issues = []
    geometries = {}

    for column in IMAGE_COLS:
        relative_path = row[column]

        if pd.isna(relative_path) or relative_path == NOT_AVAILABLE:
            continue

        path = DATA_MASTER_DIR / relative_path
        image_issues, geometry, mask_values = check_image(path, column)
        issues.extend(image_issues)

        if geometry is not None and column in DWI_COLS + TRACT_MASK_COLS:
            geometries[column] = geometry

        # Check consistency of mask formatting within each category
        if mask_values is not None:
            for group_name, columns in MASK_GROUPS.items():
                if column not in columns:
                    continue

                if group_name not in mask_formats:
                    mask_formats[group_name] = mask_values
                elif mask_values != mask_formats[group_name]:
                    issues.append(
                        f"ERROR: {column}: Mask format {mask_values} differs "
                        f"from expected {mask_formats[group_name]}"
                    )

    # Compare diffusion-derived images with FA
    if PathCols.FA in geometries:
        reference = geometries[PathCols.FA]
        for column, geometry in geometries.items():
            if column != PathCols.FA:
                issues.extend(check_geometry(geometry, reference, column))
    else:
        issues.append("WARNING: FA reference not available for geometry checks")

    if any(issue.startswith("ERROR") for issue in issues):
        status = "ERROR"
    elif issues:
        status = "WARNING"
    else:
        status = "PASS"

    results.append((subject, session, status, issues))


# %%
# Write report

report_lines.append("Mask formats:")
for group_name in MASK_GROUPS:
    format_values = mask_formats.get(group_name, "Not available")
    report_lines.append(f"  {group_name}: {format_values}")

report_lines.extend(["", "Validation summary:"])
for status in ["PASS", "WARNING", "ERROR"]:
    count = sum(result[2] == status for result in results)
    report_lines.append(f"  {status}: {count}")

report_lines.extend(["", "Results by subject/session:", ""])

for subject, session, status, issues in results:
    report_lines.append(f"sub-{subject} / ses-{session}: {status}")
    for issue in issues:
        report_lines.append(f"    {issue}")

report_path = Path(__file__).with_suffix(".txt")
report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

# %%
