"""Utility functions and objects."""

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.nifti1 import Nifti1Image
from scipy import ndimage as ndi
from skimage.graph import MCP_Geometric

from mld_mri_stats.config import DATA_MASTER_DIR, NOT_AVAILABLE


@dataclass(frozen=True)
class Cols:
    """Master table column names."""

    SUBJECT_ID: str = "SubjectID"
    INTERNAL_ID: str = "InternalID"
    INITIAL: str = "Initial"
    SESSION_ID: str = "SessionID"
    GROUP: str = "Group"
    EXCLUDED: str = "Excluded"
    EXCLUSION_REASON: str = "ExclusionReason"
    DOB: str = "DoB"
    DATE_MRI: str = "DateMRI"
    MLD_MRI_SCORE: str = "MLDMRIScore"
    IQ: str = "IQ"
    IQ_DATE: str = "IQDate"
    IQ_TEST: str = "IQTest"
    EPILEPSY: str = "Epilepsy"
    GMFC: str = "GMFC"
    MLD_FORM: str = "MLDForm"
    DATE_INITIAL_SYMPTOMS: str = "DateInitialSymptoms"
    DATE_DIAGNOSIS: str = "DateDiagnosis"
    SEX: str = "Sex"
    THERAPY_TYPE: str = "TherapyType"
    DATE_THERAPY_START: str = "DateTherapyStart"
    COMMENT: str = "Comment"
    MANUAL_MASK_SOURCE: str = "ManualMaskSource"


SNAPSHOT_COLS = [
    Cols.INTERNAL_ID,
    Cols.SESSION_ID,
    Cols.GROUP,
    Cols.MLD_MRI_SCORE,
    Cols.GMFC,
]

# labels that are to be ignored in the voronoi subparcellation of white matter;
# white mitter itself must also be ignored
WHITE_MATTER = "White_Matter"
CEREBELLUM = "Cerebellum"
BRAINSTEM = "Brainstem"
CSF = "Ventricle_CSF"
NON_REQUIRED_LABEL_STRUCTURES_WM_SUBPARCELLATION = [
    WHITE_MATTER,
    CEREBELLUM,
    BRAINSTEM,
    CSF,
]


@dataclass(frozen=True)
class BIDSTags:
    """Filename tags for image files in data master."""

    # Anatomical images
    T1: str = "_UNIT1"
    QT1: str = "_T1map"
    FLAIR: str = "_FLAIR"

    # Diffusion images
    DWI: str = "_dwi"

    # Diffusion_derivatives -  Preprocessing & Basics
    DWI_PREPROC: str = "_desc-preproc_dwi"
    BRAIN_MASK: str = "_desc-brain_mask"
    FA: str = "_FA"
    MD: str = "_MD"

    # Diffusion_derivatives -  Kurtosis Imaging
    DKI_AK: str = "_desc-AK_dwimap"
    DKI_MK: str = "_desc-MK_dwimap"
    DKI_RK: str = "_desc-RK_dwimap"

    # Diffusion_derivatives -  NODDI
    NODDI_FWF: str = "_model-NODDI_param-FWF_dwimap"
    NODDI_NDI: str = "_model-NODDI_param-NDI_dwimap"
    NODDI_ODI: str = "_model-NODDI_param-ODI_dwimap"

    # MTR
    MTR: str = "_MTR_toT1"

    # Manual white matter regions of interest
    ROI_CST_L: str = "_ROI_CST_L"
    ROI_CST_R: str = "_ROI_CST_R"
    ROI_PLIC_L: str = "_ROI_PLIC_L"
    ROI_PLIC_R: str = "_ROI_PLIC_R"
    ROI_FWM_L: str = "_ROI_FWM_L"
    ROI_FWM_R: str = "_ROI_FWM_R"


@dataclass(frozen=True)
class FibreTractMasks:
    """Names of fibre tract bundles derived from FOD."""

    CC1: str = "CCI"
    CC2: str = "CCII"
    CC3: str = "CCIII"
    CC4: str = "CCIV"
    CC5: str = "CCV"
    CST_L: str = "CST_left"
    CST_R: str = "CST_right"
    IFOF_L: str = "IFOF_left"
    IFOF_R: str = "IFOF_right"
    ILF_L: str = "ILF_left"
    ILF_R: str = "ILF_right"
    SLF1_L: str = "SLFI_left"
    SLF1_R: str = "SLFI_right"
    SLF2_L: str = "SLFII_left"
    SLF2_R: str = "SLFII_right"
    SLF3_L: str = "SLFIII_left"
    SLF3_R: str = "SLFIII_right"


@dataclass(frozen=True)
class PathCols:
    """Master table column names for imaging file paths."""

    # Anatomical images
    T1: str = "PathT1"
    QT1: str = "PathQT1"
    FLAIR: str = "PathFLAIR"

    # Diffusion - DTI
    FA: str = "PathFA"
    MD: str = "PathMD"

    # Diffusion - DKI
    DKI_AK: str = "PathDKI_AK"
    DKI_MK: str = "PathDKI_MK"
    DKI_RK: str = "PathDKI_RK"

    # Diffusion - NODDI
    NODDI_FWF: str = "PathNODDI_FWF"
    NODDI_NDI: str = "PathNODDI_NDI"
    NODDI_ODI: str = "PathNODDI_ODI"

    # Diffusion - FOD
    FOD_WM_NORM: str = "PathFODWMNorm"
    FOD_INTEGRAL: str = "PathFODIntegral"

    # MTR
    MTR: str = "PathMTR"

    # Manual white matter ROIs
    ROI_CST_L: str = "PathROI_CST_L"
    ROI_CST_R: str = "PathROI_CST_R"
    ROI_PLIC_L: str = "PathROI_PLIC_L"
    ROI_PLIC_R: str = "PathROI_PLIC_R"
    ROI_FWM_L: str = "PathROI_FWM_L"
    ROI_FWM_R: str = "PathROI_FWM_R"

    # FOD-derived fibre tract masks
    TRACT_CC1: str = "PathTract_CC1"
    TRACT_CC2: str = "PathTract_CC2"
    TRACT_CC3: str = "PathTract_CC3"
    TRACT_CC4: str = "PathTract_CC4"
    TRACT_CC5: str = "PathTract_CC5"
    TRACT_CST_L: str = "PathTract_CST_L"
    TRACT_CST_R: str = "PathTract_CST_R"
    TRACT_IFOF_L: str = "PathTract_IFOF_L"
    TRACT_IFOF_R: str = "PathTract_IFOF_R"
    TRACT_ILF_L: str = "PathTract_ILF_L"
    TRACT_ILF_R: str = "PathTract_ILF_R"
    TRACT_SLF1_L: str = "PathTract_SLF1_L"
    TRACT_SLF1_R: str = "PathTract_SLF1_R"
    TRACT_SLF2_L: str = "PathTract_SLF2_L"
    TRACT_SLF2_R: str = "PathTract_SLF2_R"
    TRACT_SLF3_L: str = "PathTract_SLF3_L"
    TRACT_SLF3_R: str = "PathTract_SLF3_R"


def load_nifti(path: str | Path, canonical: bool = True) -> Nifti1Image:
    """Load NIFTI and reorient to RAS+ orientation.

    Args:
        path: Path to Nifti image.
        canonical: Set if reorientation to RAS desired. Defaults to true.

    """
    img = nib.load(str(path))
    if not isinstance(img, Nifti1Image):
        raise TypeError("Unexpected image type")
    if canonical:
        img = nib.as_closest_canonical(img)
    return img


def save_nifti(nifti: Nifti1Image, outname: Path) -> None:
    """Save NIFTI Wrapper to silence Pylance warnings.

    Args:
        nifti: Nifti image.
        outname: Out path.

    """
    nib.save(nifti, outname)


def validate_integer_score(
    df: pd.DataFrame, col: str, expected_range: list[int]
) -> None:
    """Validate a pd df column to have integer scores within the expected range."""
    values = df[col]
    lower, upper = expected_range

    if values.isna().any():
        raise ValueError(f"{col} contains missing values.")

    if not values.mod(1).eq(0).all():
        raise ValueError(f"{col} must contain integers.")

    out_of_range = ~values.between(lower, upper)
    if out_of_range.any():
        raise ValueError(
            f"{col} contains values outside the expected range "
            f"[{lower}, {upper}]: {sorted(values[out_of_range].unique())}"
        )


def find_unique_path(
    paths: list[Path],
    subject_id: str,
    session_id: str,
    tag: str,
) -> str:
    """Find a unique file matching subject, session, and modality tag."""
    matches = [
        p
        for p in paths
        if f"sub-{subject_id}" in p.parts
        and f"ses-{session_id}" in p.parts
        and tag in p.name
    ]

    if len(matches) > 1:
        raise ValueError(
            f"Multiple matches for {subject_id}, {session_id}, {tag}: {matches}"
        )

    return str(matches[0].relative_to(DATA_MASTER_DIR)) if matches else NOT_AVAILABLE


def merge_hemisphere_labelmaps(
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    """Merge non-overlapping left and right hemisphere label maps.

    Combines two integer label arrays into a single array, preserving all
    non-zero labels. Zero represents background. Raises an error if the
    arrays have different shapes, non-integer types, or overlapping labels.

    """
    if left.shape != right.shape:
        raise ValueError("Shapes must match.")
    if not np.issubdtype(left.dtype, np.integer) or not np.issubdtype(
        right.dtype, np.integer
    ):
        raise TypeError("Inputs should be integer label images.")

    lmask = left != 0
    rmask = right != 0
    overlap = lmask & rmask
    n_overlap = int(overlap.sum())

    if n_overlap:
        raise ValueError(f"Left/right overlap at {n_overlap} voxels.")

    return np.where(lmask, left, right)


def geodesic_subparcellate(
    wm_mask: np.ndarray,
    seed_labels: np.ndarray,
    spacing: tuple[float, ...],
) -> np.ndarray:
    """Parcellate white matter by shortest paths to anatomical seed regions.

    Computes geodesic distances through white matter using 26-connectivity.
    The WM mask is dilated by one voxel to include bordering anatomical seeds.
    Other tissues, including CSF, remain impassable.

    Parameters
    ----------
    wm_mask : np.ndarray
        Boolean mask of the white matter to parcellate.
    seed_labels : np.ndarray
        Anatomical label image. Zero represents background.
    spacing : tuple of float
        Voxel dimensions in mm, in array axis order.

    Returns
    -------
    np.ndarray
        Integer label image containing the nearest anatomical seed label
        for each reachable WM voxel. Other voxels remain zero.

    """
    # Define traversable voxels: WM and immediately adjacent seeds.
    dilated = ndi.binary_dilation(wm_mask)
    seed_mask = dilated & (seed_labels != 0)
    allowed = wm_mask | seed_mask

    costs = np.where(allowed, 1.0, np.inf)
    starts = np.argwhere(seed_mask)

    # Compute shortest paths from all seeds.
    mcp = MCP_Geometric(costs, fully_connected=True, sampling=spacing)
    _, traceback = mcp.find_costs(starts=starts)

    # Construct a map pointing each voxel to its predecessor.
    parents = np.arange(seed_labels.size, dtype=np.int32).reshape(seed_labels.shape)

    for direction, offset in enumerate(mcp.offsets):
        coords = np.where(traceback == direction)
        predecessor = tuple(
            coord - step for coord, step in zip(coords, offset, strict=True)
        )
        parents[coords] = np.ravel_multi_index(predecessor, seed_labels.shape)

    # Follow predecessors until all voxels point to their originating seed.
    roots = parents.ravel()

    while True:
        next_roots = roots[roots]
        if np.array_equal(next_roots, roots):
            break
        roots = next_roots

    # Assign the originating seed labels to WM voxels.
    nearest_labels = seed_labels.ravel()[roots].reshape(seed_labels.shape)
    result = np.zeros(seed_labels.shape, dtype=np.int32)
    result[wm_mask] = nearest_labels[wm_mask]

    return result


def median_range(series):
    """Format median and range."""
    series = series.dropna()

    if series.empty:
        return "–"

    return f"{series.median():.1f} ({series.min():.1f}–{series.max():.1f})"


def median_iqr_range(series):
    """Format median, interquartile interval and range."""
    series = series.dropna()

    if series.empty:
        return "–"

    return (
        f"{series.median():.1f} "
        f"[{series.quantile(0.25):.1f}–{series.quantile(0.75):.1f}]; "
        f"({series.min():.1f}–{series.max():.1f})"
    )


def category_counts(series):
    """Format counts of unique categories."""
    counts = series.value_counts()

    if counts.empty:
        return "–"

    return ", ".join(f"{category}: {n}" for category, n in counts.items())


def describe_cohort(df):
    """Calculate subject-level and session-level descriptive statistics.

    Categorical characteristics are summarised at subject level, based on
    the first included MRI session. Age and clinical scores are summarised
    at session level. Missing values are omitted.
    """
    subjects = df.sort_values(Cols.DATE_MRI).drop_duplicates(Cols.SUBJECT_ID)
    sessions_per_subject = df.groupby(Cols.SUBJECT_ID).size()

    return {
        "Total N sessions": len(df),
        "Total N subjects": df[Cols.SUBJECT_ID].nunique(),
        "Sessions per subject": median_range(sessions_per_subject),
        "Age at MRI (years)": median_iqr_range(df["AgeMRI"]),
        "MLD MRI score": median_iqr_range(df[Cols.MLD_MRI_SCORE]),
        "GMFC": median_iqr_range(df[Cols.GMFC]),
        "MLD form": category_counts(subjects[Cols.MLD_FORM]),
        "Sex": category_counts(subjects[Cols.SEX]),
        "Therapy type": category_counts(subjects[Cols.THERAPY_TYPE]),
    }
