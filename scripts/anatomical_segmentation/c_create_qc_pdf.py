"""Create a QC PDF for SynthSeg segmentation and T1-to-FA registration.

Each page shows:
- Three axial T1 slices with transparent FA overlay
- Three axial skull-stripped T1 slices with SynthSeg labels

ANTs transformations are applied in memory.

Output:
- PDF report in the repository's reports folder
"""

# %%

import ants
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec

from mld_mri_stats.config import (
    DATA_MASTER_DIR,
    MASTER_DATA_CSV,
    REPO_ROOT,
    TEMPORARY_DATA_DIR,
)
from mld_mri_stats.utils import Cols, PathCols

# Paths
SYNTHSEG_DIR = TEMPORARY_DATA_DIR / "synthseg"
REG_DIR = TEMPORARY_DATA_DIR / "registration_T1_to_FA"
QC_PDF_PATH = REPO_ROOT / "reports" / "qc_synthseg_and_t1_fa_registration.pdf"

# Visualization settings (from previous QC script)
FA_ALPHA = 0.7
SEGMENTATION_ALPHA = 0.6
AXIAL_SLICES_FRAC = (0.30, 0.50, 0.70)
FIGSIZE = (11.7, 8.3)  # A4 landscape
FA_HIDE_CUTOFF = 0.15

# %%
# define helper functions


def pick_axial_indices(shape: tuple, fracs: tuple) -> list[int]:
    """Pick axial slice indices at relative positions through the volume."""
    z = shape[-1]
    return [int(round(frac * (z - 1))) for frac in fracs]


def relabel_to_ordinal(labels: np.ndarray) -> np.ndarray:
    """Map segmentation labels to consecutive integers, preserving background 0."""
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != 0]

    result = np.zeros(labels.shape, dtype=np.int32)

    for i, value in enumerate(unique_labels, start=1):
        result[labels == value] = i

    return result


def make_discrete_cmap(n_classes: int) -> ListedColormap:
    """Create a discrete colormap with transparent background."""
    base = plt.get_cmap("tab20")
    colors = [(0.0, 0.0, 0.0, 0.0)]

    for i in range(n_classes):
        colors.append(base(i % base.N))

    return ListedColormap(colors)


def apply_transform(
    fixed: ants.ANTsImage,
    moving: ants.ANTsImage,
    transform_path: str,
    interpolator: str = "linear",
) -> np.ndarray:
    """Transform an image into the fixed image space."""
    moved = ants.apply_transforms(
        fixed=fixed,
        moving=moving,
        transformlist=[transform_path],
        interpolator=interpolator,
    )
    return moved.numpy()  # type: ignore


# %%
# Read master table

main_data_df = pd.read_csv(
    MASTER_DATA_CSV,
    sep=";",
    dtype={Cols.SUBJECT_ID: str, Cols.SESSION_ID: str},
)

QC_PDF_PATH.parent.mkdir(parents=True, exist_ok=True)

# %%
# Create PDF

with PdfPages(QC_PDF_PATH) as pdf:
    for _, row in main_data_df.iterrows():
        subject = row[Cols.SUBJECT_ID]
        session = row[Cols.SESSION_ID]

        base = f"sub-{subject}_ses-{session}"

        # Input paths
        t1_path = DATA_MASTER_DIR / row[PathCols.T1]
        fa_path = DATA_MASTER_DIR / row[PathCols.FA]

        t1_brain_path = SYNTHSEG_DIR / f"{base}_brain.nii.gz"
        synthseg_path = SYNTHSEG_DIR / f"{base}_synthseg_labels.nii.gz"
        transform_path = REG_DIR / f"{base}_T1_to_FA_0GenericAffine.mat"

        print(f"Processing {base}")

        # Load images with ANTsPy
        fa_img = ants.image_read(str(fa_path))
        t1_img = ants.image_read(str(t1_path))
        brain_img = ants.image_read(str(t1_brain_path))
        seg_img = ants.image_read(str(synthseg_path))

        fa = fa_img.numpy()

        # Transform T1 images and segmentation into FA space
        t1 = apply_transform(fa_img, t1_img, str(transform_path))
        brain = apply_transform(fa_img, brain_img, str(transform_path))
        seg = apply_transform(
            fa_img,
            seg_img,
            str(transform_path),
            interpolator="genericLabel",
        )

        # Prepare FA overlay: hide low FA values
        fa_mask = fa > FA_HIDE_CUTOFF
        fa_display = np.ma.masked_where(~fa_mask, fa)

        fa_vmin, fa_vmax = np.percentile(fa[fa_mask], [1, 99])

        # Prepare segmentation overlay
        seg = relabel_to_ordinal(seg)
        n_classes = int(seg.max())
        seg_cmap = make_discrete_cmap(n_classes)

        # Image intensity ranges
        t1_vmin, t1_vmax = float(t1.min()), float(t1.max())
        brain_vmin, brain_vmax = float(brain.min()), float(brain.max())

        # Choose slices
        z_indices = pick_axial_indices(fa.shape, AXIAL_SLICES_FRAC)

        # Create figure
        fig = plt.figure(figsize=FIGSIZE, constrained_layout=True)
        gs = GridSpec(2, 3, figure=fig, hspace=0.06, wspace=0.02)

        fig.suptitle(f"SUBJECT: {subject}    SESSION: {session}", fontsize=14)

        # Row 1: T1 + FA overlay
        for j, z in enumerate(z_indices):
            ax = fig.add_subplot(gs[0, j])

            ax.imshow(
                np.rot90(t1[:, :, z]),
                cmap="gray",
                vmin=t1_vmin,
                vmax=t1_vmax,
            )
            ax.imshow(
                np.rot90(fa_display[:, :, z]),
                cmap="inferno",
                vmin=fa_vmin,
                vmax=fa_vmax,
                alpha=FA_ALPHA,
            )

            ax.set_axis_off()

            if j == 0:
                ax.set_title("T1 + FA overlay", fontsize=11, loc="left")

        # Row 2: Skull-stripped T1 + SynthSeg labels
        for j, z in enumerate(z_indices):
            ax = fig.add_subplot(gs[1, j])

            ax.imshow(
                np.rot90(brain[:, :, z]),
                cmap="gray",
                vmin=brain_vmin,
                vmax=brain_vmax,
            )
            ax.imshow(
                np.rot90(seg[:, :, z]),
                cmap=seg_cmap,
                vmin=0,
                vmax=n_classes,
                alpha=SEGMENTATION_ALPHA,
            )

            ax.set_axis_off()

            if j == 0:
                ax.set_title(
                    "Skull-stripped T1 + SynthSeg labels",
                    fontsize=11,
                    loc="left",
                )

        pdf.savefig(fig, dpi=200)
        plt.close(fig)

print(f"QC PDF written to: {QC_PDF_PATH}")

# %%
