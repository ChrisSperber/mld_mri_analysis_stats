"""Utility functions and objects."""

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
from nibabel.nifti1 import Nifti1Image


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


@dataclass(frozen=True)
class BIDSTags:
    """Filename tags for image files in data master."""

    # Anatomical images
    T1: str = "_UNIT1"
    QT1: str = "_T1map"
    FLAIR: str = "_FLAIR"

    # Diffusion images & preprocessing
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


def load_nifti(path: str | Path) -> Nifti1Image:
    """Load NIFTI Wrapper with typechecking to silence Pylance warnings."""
    img = nib.load(str(path))
    if not isinstance(img, Nifti1Image):
        raise TypeError("Unexpected image type")
    return img


def save_nifti(nifti: Nifti1Image, outname: Path) -> None:
    """Save NIFTI Wrapper to silence Pylance warnings.

    Args:
        nifti: Nifti image.
        outname: Out path.

    """
    nib.save(nifti, outname)
