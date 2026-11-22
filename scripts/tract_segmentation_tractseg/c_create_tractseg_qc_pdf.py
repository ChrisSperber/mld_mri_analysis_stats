"""Generate QC PDFs for TractSeg outputs.

Each page shows one subject/session for one tract group:
- Row 1: binary tract mask on FOD integral background
- Row 2: probability map binned into >0.20, >0.35, >0.50

Behavior:
- Tract groups are combined across multiple masks (e.g. left + right)
- Binary masks are combined by logical OR
- Probability maps are combined by voxelwise maximum
- Slice indices are chosen from occupied slices of the tract mask
- Coronal view is used for selected tract groups such as CST

Output:
- One PDF per tract group, written in repo root/reports
"""

# %%

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from nibabel.nifti1 import Nifti1Image
from nilearn.image import resample_to_img

from mld_mri_stats.config import (
    DATA_MASTER_DIR,
    MASTER_DATA_CSV,
    TEMPORARY_DATA_DIR,
)
from mld_mri_stats.utils import Cols, PathCols, load_nifti

# %%
TRACTSEG_DIR = TEMPORARY_DATA_DIR / "tractseg"
OUT_SUBFOLDER = Path(__file__).parents[2] / "reports" / "tractseg_qc_pdfs"

# Edit these tract groups as desired
QC_TRACTS: list[list[str]] = [
    ["CST_left", "CST_right"],
    ["SLF_III_left", "SLF_III_right"],
]

BINARY_SUBDIR = "bundle_segmentations"
PROB_SUBDIR = "bundle_probabilities"

CORONAL_VIS_TRACTS = ["CST", "FPT"]

N_SLICES = 3
FIGSIZE = (11.7, 8.3)  # A4 landscape
MASK_ALPHA = 0.55

BG_WINDOW_PCT = (1.0, 99.0)

PROB_THRESHOLD_LOW = 0.20
PROB_THRESHOLD_MID = 0.35
PROB_THRESHOLD_HIGH = 0.50

# %%


def _load_nifti(path: Path) -> Nifti1Image:
    """Load NIfTI using the shared utility wrapper."""
    return load_nifti(path)


def _binary_mask_paths(subject_dir: Path, tract_names: Sequence[str]) -> list[Path]:
    """Return binary tract mask paths for one subject/session."""
    seg_dir = subject_dir / BINARY_SUBDIR
    return [seg_dir / f"{tract}.nii.gz" for tract in tract_names]


def _probability_paths(subject_dir: Path, tract_names: Sequence[str]) -> list[Path]:
    """Return tract probability map paths for one subject/session."""
    prob_dir = subject_dir / PROB_SUBDIR
    return [prob_dir / f"{tract}.nii.gz" for tract in tract_names]


def _to_bool_mask(img: Nifti1Image) -> np.ndarray:
    """Convert NIfTI image to boolean mask."""
    data = np.asarray(img.get_fdata(), dtype=np.float32)
    return data > 0


def _resample_mask_to_bg(mask_img: Nifti1Image, bg_img: Nifti1Image) -> np.ndarray:
    """Resample binary mask to background image grid."""
    mask_r: Nifti1Image = resample_to_img(
        mask_img,
        bg_img,
        interpolation="nearest",
        force_resample=True,
        copy_header=True,
    )  # pyright: ignore[reportAssignmentType]
    return _to_bool_mask(mask_r)


def _resample_prob_to_bg(prob_img: Nifti1Image, bg_img: Nifti1Image) -> np.ndarray:
    """Resample probability image to background image grid."""
    prob_r: Nifti1Image = resample_to_img(
        prob_img,
        bg_img,
        interpolation="continuous",
        force_resample=True,
        copy_header=True,
    )  # pyright: ignore[reportAssignmentType]
    data = np.asarray(prob_r.get_fdata(), dtype=np.float32)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(data, 0.0, 1.0)


def _pick_slices(mask: np.ndarray, axis: int, n_slices: int) -> list[int]:
    """Pick slice indices using quantiles over occupied slices."""
    if mask.ndim != 3:  # noqa: PLR2004
        raise ValueError(f"Expected 3D mask, got shape {mask.shape}")

    other_axes = tuple(ax for ax in (0, 1, 2) if ax != axis)
    idx_any = np.where(mask.any(axis=other_axes))[0]

    if idx_any.size == 0:
        return []

    if idx_any.size == 1:
        return [int(idx_any[0])] * n_slices

    qs = np.linspace(0.15, 0.85, n_slices) if n_slices > 1 else np.array([0.5])
    idx_q = np.quantile(idx_any.astype(np.float32), qs, method="nearest")
    idx = sorted({int(i) for i in idx_q})

    if len(idx) != n_slices:
        i_min, i_max = int(idx_any.min()), int(idx_any.max())
        if i_min == i_max:
            return [i_min] * n_slices
        i_lin = np.linspace(i_min, i_max, n_slices)
        return [int(round(i)) for i in i_lin]

    return idx


def _should_use_coronal_view(tract_group: Sequence[str]) -> bool:
    """Use coronal view for configured tracts."""
    return any(tract.startswith(tuple(CORONAL_VIS_TRACTS)) for tract in tract_group)


def _extract_slice(arr: np.ndarray, axis: int, idx: int) -> np.ndarray:
    """Extract a 2D slice from a 3D array."""
    if axis == 0:
        return arr[idx, :, :]
    if axis == 1:
        return arr[:, idx, :]
    return arr[:, :, idx]


def _axis_label(axis: int) -> str:
    """Return axis label."""
    return {0: "x", 1: "y", 2: "z"}[axis]


def _rgba_red_overlay(mask2d: np.ndarray, alpha: float) -> np.ndarray:
    """RGBA overlay: red for True voxels."""
    m = mask2d.astype(np.float32, copy=False)
    overlay = np.zeros((m.shape[0], m.shape[1], 4), dtype=np.float32)
    overlay[..., 0] = 1.0
    overlay[..., 3] = alpha * m
    return overlay


def _rgba_probability_overlay(prob2d: np.ndarray) -> np.ndarray:
    """RGBA overlay for binned probability map.

    Bins:
    - > 0.20 and <= 0.35: yellow
    - > 0.35 and <= 0.50: orange
    - > 0.50: red
    """
    overlay = np.zeros((prob2d.shape[0], prob2d.shape[1], 4), dtype=np.float32)

    low = (prob2d > PROB_THRESHOLD_LOW) & (prob2d <= PROB_THRESHOLD_MID)
    mid = (prob2d > PROB_THRESHOLD_MID) & (prob2d <= PROB_THRESHOLD_HIGH)
    high = prob2d > PROB_THRESHOLD_HIGH

    # yellow
    overlay[low, 0] = 1.0
    overlay[low, 1] = 1.0
    overlay[low, 3] = 0.30

    # orange
    overlay[mid, 0] = 1.0
    overlay[mid, 1] = 0.55
    overlay[mid, 3] = 0.45

    # red
    overlay[high, 0] = 1.0
    overlay[high, 3] = 0.60

    return overlay


def _placeholder_page(pdf: PdfPages, title: str, lines: list[str]) -> None:
    """Create a placeholder page with text."""
    fig = plt.figure(figsize=FIGSIZE, constrained_layout=True)
    fig.suptitle(title, fontsize=14)

    ax = fig.add_subplot(1, 1, 1)
    ax.set_axis_off()
    ax.text(
        0.02,
        0.95,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=12,
        family="monospace",
        transform=ax.transAxes,
    )

    pdf.savefig(fig, dpi=200)
    plt.close(fig)


# %%
OUT_SUBFOLDER.mkdir(parents=True, exist_ok=True)

main_data_df = pd.read_csv(
    MASTER_DATA_CSV,
    sep=";",
    dtype={Cols.SUBJECT_ID: str, Cols.SESSION_ID: str},
)

# %%
# Create one PDF per tract group

for tract_group in QC_TRACTS:
    tract_label = "+".join(tract_group)
    tract_label_safe = tract_label.replace("/", "_").replace(" ", "_")
    out_pdf = OUT_SUBFOLDER / f"qc_{tract_label_safe}.pdf"

    view_axis = 1 if _should_use_coronal_view(tract_group) else 2
    view_axis_name = _axis_label(view_axis)

    print(
        f"\n=== Writing QC PDF for tract-group: {tract_label} (view={view_axis_name})"
    )
    print(f"-> {out_pdf}")

    with PdfPages(out_pdf) as pdf:
        for _, row in main_data_df.iterrows():
            subject = row[Cols.SUBJECT_ID]
            session = row[Cols.SESSION_ID]
            base = f"sub-{subject}_ses-{session}"

            title = f"SUBJECT: {base}    TRACT: {tract_label}"

            bg_path = DATA_MASTER_DIR / row[PathCols.FOD_INTEGRAL]
            subject_dir = TRACTSEG_DIR / base

            binary_paths = _binary_mask_paths(subject_dir, tract_group)
            prob_paths = _probability_paths(subject_dir, tract_group)

            missing: list[str] = []

            if not bg_path.exists():
                missing.append(f"FOD integral missing: {bg_path.name}")

            for path in binary_paths:
                if not path.exists():
                    missing.append(f"Binary mask missing: {path.name}")

            for path in prob_paths:
                if not path.exists():
                    missing.append(f"Probability map missing: {path.name}")

            if missing:
                _placeholder_page(
                    pdf,
                    title=title,
                    lines=["Missing inputs -> no overlay plotted.", "", *missing],
                )
                continue

            # Load background
            bg_img = _load_nifti(bg_path)
            bg = np.asarray(bg_img.get_fdata(), dtype=np.float32)

            if bg.ndim != 3:  # noqa: PLR2004
                _placeholder_page(
                    pdf,
                    title=title,
                    lines=[f"Unexpected background shape: {bg.shape} (expected 3D)."],
                )
                continue

            finite = np.isfinite(bg)
            if finite.any():
                vmin, vmax = np.percentile(bg[finite], BG_WINDOW_PCT)
                vmin_bg, vmax_bg = float(vmin), float(vmax)
            else:
                vmin_bg, vmax_bg = 0.0, 1.0

            # Combine binary masks
            combined_binary = np.zeros(bg.shape, dtype=bool)

            for path in binary_paths:
                mask_img = _load_nifti(path)
                mask_bool = _resample_mask_to_bg(mask_img, bg_img)

                if mask_bool.shape != combined_binary.shape:
                    _placeholder_page(
                        pdf,
                        title=title,
                        lines=[
                            "Binary mask resampling produced unexpected shape.",
                            f"Background shape: {combined_binary.shape}",
                            f"Mask shape:       {mask_bool.shape}",
                            f"Mask file:        {path.name}",
                        ],
                    )
                    combined_binary = None  # type: ignore[assignment]
                    break

                combined_binary |= mask_bool

            if combined_binary is None:
                continue

            # Combine probability maps by voxelwise maximum
            combined_prob = np.zeros(bg.shape, dtype=np.float32)

            for path in prob_paths:
                prob_img = _load_nifti(path)
                prob = _resample_prob_to_bg(prob_img, bg_img)

                if prob.shape != combined_prob.shape:
                    _placeholder_page(
                        pdf,
                        title=title,
                        lines=[
                            "Probability resampling produced unexpected shape.",
                            f"Background shape: {combined_prob.shape}",
                            f"Prob shape:       {prob.shape}",
                            f"Prob file:        {path.name}",
                        ],
                    )
                    combined_prob = None  # type: ignore[assignment]
                    break

                combined_prob = np.maximum(combined_prob, prob)

            if combined_prob is None:
                continue

            # Determine occupied region for slice picking
            occupancy = combined_binary.copy()
            if not occupancy.any():
                occupancy = combined_prob > PROB_THRESHOLD_LOW

            if not occupancy.any():
                _placeholder_page(
                    pdf,
                    title=title,
                    lines=[
                        "Combined tract is empty.",
                        "",
                        "Binary masks and probabilities had no occupied voxels.",
                    ],
                )
                continue

            slice_idxs = _pick_slices(occupancy, axis=view_axis, n_slices=N_SLICES)
            if not slice_idxs:
                _placeholder_page(
                    pdf,
                    title=title,
                    lines=["Could not determine slice indices."],
                )
                continue

            # Plot 2 rows × N_SLICES
            fig = plt.figure(figsize=FIGSIZE, constrained_layout=True)
            gs = GridSpec(2, len(slice_idxs), figure=fig, hspace=0.04, wspace=0.02)

            fig.suptitle(
                f"{title}    VIEW: {view_axis_name}\n"
                f"Probabilities: yellow > {PROB_THRESHOLD_LOW}, "
                f"orange > {PROB_THRESHOLD_MID}, red > {PROB_THRESHOLD_HIGH}",
                fontsize=13,
            )

            # Row 1: binary segmentation
            for j, idx in enumerate(slice_idxs):
                ax = fig.add_subplot(gs[0, j])

                bg2d = np.rot90(_extract_slice(bg, axis=view_axis, idx=idx))
                binary2d = np.rot90(
                    _extract_slice(combined_binary, axis=view_axis, idx=idx)
                )

                ax.imshow(bg2d, cmap="gray", vmin=vmin_bg, vmax=vmax_bg)
                ax.imshow(_rgba_red_overlay(binary2d, MASK_ALPHA))
                ax.set_axis_off()

                if j == 0:
                    ax.set_title(
                        f"Binary mask    {view_axis_name}={idx}",
                        fontsize=11,
                        loc="left",
                    )
                else:
                    ax.set_title(f"{view_axis_name}={idx}", fontsize=11)

            # Row 2: binned probabilities
            for j, idx in enumerate(slice_idxs):
                ax = fig.add_subplot(gs[1, j])

                bg2d = np.rot90(_extract_slice(bg, axis=view_axis, idx=idx))
                prob2d = np.rot90(
                    _extract_slice(combined_prob, axis=view_axis, idx=idx)
                )

                ax.imshow(bg2d, cmap="gray", vmin=vmin_bg, vmax=vmax_bg)
                ax.imshow(_rgba_probability_overlay(prob2d))
                ax.set_axis_off()

                if j == 0:
                    ax.set_title(
                        f"Probability bins    {view_axis_name}={idx}",
                        fontsize=11,
                        loc="left",
                    )
                else:
                    ax.set_title(f"{view_axis_name}={idx}", fontsize=11)

            pdf.savefig(fig, dpi=200)
            plt.close(fig)

    print(f"Done: {out_pdf}")

# %%
