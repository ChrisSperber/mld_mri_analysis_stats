"""Fetch, verify, and copy main data and collect file paths.

Collect paths to all relevant imaging files and derivatives relative to DATA_MASTER_DIR. Missing
data are marked as NOT_AVAILABLE. Data are merged with the main table and locally stored for use in
downstream analyses.
Data format and range of main variables (GMFC, MRI Score) is also verified.

NOTE: The output of this script is a requirement for all further analyses in this repo.

Output:
- a copy of the data master table in the current repo as CSV including paths to image files
    (to be excluded from git)
- a minimalistic copy of the data master table in the current repo as CSV with all sensitive
    information redacted with the purpose of documentation
- a small overview txt giving total numbers of patients per modality
"""

# %%
from pathlib import Path

import pandas as pd

from mld_mri_stats.config import (
    DATA_MASTER_DIR,
    MASTER_DATA_CSV,
    MASTER_DATA_CSV_SNAPSHOT,
    NOT_AVAILABLE,
    SOURCE_MASTER_XLS,
)
from mld_mri_stats.utils import (
    SNAPSHOT_COLS,
    BIDSTags,
    Cols,
    FibreTractMasks,
    PathCols,
    find_unique_path,
    validate_integer_score,
)

EXPECTED_RANGE_GMFC = [0, 6]
EXPECTED_RANGE_MRISCORE = [0, 34]

FILE_TAGS = {
    # Anatomical images
    PathCols.T1: BIDSTags.T1,
    PathCols.QT1: BIDSTags.QT1,
    PathCols.FLAIR: BIDSTags.FLAIR,
    # Diffusion - DTI
    PathCols.FA: BIDSTags.FA,
    PathCols.MD: BIDSTags.MD,
    # Diffusion - DKI
    PathCols.DKI_AK: BIDSTags.DKI_AK,
    PathCols.DKI_MK: BIDSTags.DKI_MK,
    PathCols.DKI_RK: BIDSTags.DKI_RK,
    # Diffusion - NODDI
    PathCols.NODDI_FWF: BIDSTags.NODDI_FWF,
    PathCols.NODDI_NDI: BIDSTags.NODDI_NDI,
    PathCols.NODDI_ODI: BIDSTags.NODDI_ODI,
    # Diffusion - FOD
    PathCols.FOD_WM_NORM: "FOD_wm_norm.mif",
    PathCols.FOD_INTEGRAL: "FOD_wm_integral.nii.gz",
    # MTR
    PathCols.MTR: BIDSTags.MTR,
    # Manual white matter ROIs
    PathCols.ROI_CST_L: BIDSTags.ROI_CST_L,
    PathCols.ROI_CST_R: BIDSTags.ROI_CST_R,
    PathCols.ROI_PLIC_L: BIDSTags.ROI_PLIC_L,
    PathCols.ROI_PLIC_R: BIDSTags.ROI_PLIC_R,
    PathCols.ROI_FWM_L: BIDSTags.ROI_FWM_L,
    PathCols.ROI_FWM_R: BIDSTags.ROI_FWM_R,
    # FOD-derived fibre tract masks
    PathCols.TRACT_CC1: f"{FibreTractMasks.CC1}_mask.nii.gz",
    PathCols.TRACT_CC2: f"{FibreTractMasks.CC2}_mask.nii.gz",
    PathCols.TRACT_CC3: f"{FibreTractMasks.CC3}_mask.nii.gz",
    PathCols.TRACT_CC4: f"{FibreTractMasks.CC4}_mask.nii.gz",
    PathCols.TRACT_CC5: f"{FibreTractMasks.CC5}_mask.nii.gz",
    PathCols.TRACT_CST_L: f"{FibreTractMasks.CST_L}_mask.nii.gz",
    PathCols.TRACT_CST_R: f"{FibreTractMasks.CST_R}_mask.nii.gz",
    PathCols.TRACT_IFOF_L: f"{FibreTractMasks.IFOF_L}_mask.nii.gz",
    PathCols.TRACT_IFOF_R: f"{FibreTractMasks.IFOF_R}_mask.nii.gz",
    PathCols.TRACT_ILF_L: f"{FibreTractMasks.ILF_L}_mask.nii.gz",
    PathCols.TRACT_ILF_R: f"{FibreTractMasks.ILF_R}_mask.nii.gz",
    PathCols.TRACT_SLF1_L: f"{FibreTractMasks.SLF1_L}_mask.nii.gz",
    PathCols.TRACT_SLF1_R: f"{FibreTractMasks.SLF1_R}_mask.nii.gz",
    PathCols.TRACT_SLF2_L: f"{FibreTractMasks.SLF2_L}_mask.nii.gz",
    PathCols.TRACT_SLF2_R: f"{FibreTractMasks.SLF2_R}_mask.nii.gz",
    PathCols.TRACT_SLF3_L: f"{FibreTractMasks.SLF3_L}_mask.nii.gz",
    PathCols.TRACT_SLF3_R: f"{FibreTractMasks.SLF3_R}_mask.nii.gz",
}

# %%
# read source data table and remove excluded subjects
main_data_df = pd.read_excel(SOURCE_MASTER_XLS)

main_data_df = main_data_df[main_data_df[Cols.EXCLUDED] == 0]
main_data_df = main_data_df.drop(
    [Cols.EXCLUDED, Cols.EXCLUSION_REASON, Cols.COMMENT], axis=1
)

# Validate main variables GMFC, MLD MRI score being without missing and integers within the expected
# range
validate_integer_score(main_data_df, Cols.GMFC, EXPECTED_RANGE_GMFC)
validate_integer_score(
    main_data_df[main_data_df[Cols.GROUP] == "patient"],
    Cols.MLD_MRI_SCORE,
    EXPECTED_RANGE_MRISCORE,
)

# %%
# create minimalistic copy with all sensitive information redacted and store snapshot
snapshot_copy_df = main_data_df[SNAPSHOT_COLS].copy()
MASTER_DATA_CSV_SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
snapshot_copy_df.to_csv(MASTER_DATA_CSV_SNAPSHOT, sep=";", index=False)

# %%
# collect paths to imaging files

all_paths = [
    file
    for file in DATA_MASTER_DIR.rglob("*")
    if file.is_file()
    and file.name.endswith(".mif")
    or file.name.endswith(".nii.gz")
    or file.name.endswith(".nii")
]

for column, tag in FILE_TAGS.items():
    main_data_df[column] = [
        find_unique_path(all_paths, str(sub), str(ses), tag)
        for sub, ses in zip(
            main_data_df[Cols.SUBJECT_ID],
            main_data_df[Cols.SESSION_ID],
            strict=True,
        )
    ]

# %%
# store main df
# WARNING: file contains confidential data and must NOT be committed to the repo
MASTER_DATA_CSV.parent.mkdir(parents=True, exist_ok=True)
main_data_df.to_csv(MASTER_DATA_CSV, sep=";")

# %%
# Summarize imaging data availability for patients

patient_df = main_data_df[main_data_df[Cols.GROUP] == "patient"]

SUMMARY_COLS = {
    "T1": PathCols.T1,
    "qT1": PathCols.QT1,
    "FLAIR": PathCols.FLAIR,
    "MTR": PathCols.MTR,
    "FA": PathCols.FA,
    "FOD integral": PathCols.FOD_INTEGRAL,
    "FOD registered": PathCols.TRACT_CC1,
}

# Count sessions per patient
sessions_per_patient = patient_df.groupby(Cols.SUBJECT_ID)[Cols.SESSION_ID].nunique()
session_distribution = sessions_per_patient.value_counts().sort_index()

# Build report
report_lines = [
    "Imaging data availability - patients",
    f"Total patients: {len(sessions_per_patient)}",
    f"Total sessions: {len(patient_df)}",
    "",
    "Available imaging files (sessions):",
]

for name, column in SUMMARY_COLS.items():
    n_available = (patient_df[column] != NOT_AVAILABLE).sum()
    report_lines.append(f"{name:<16} {n_available:>3} / {len(patient_df)}")

report_lines.extend(
    [
        "",
        "Sessions per patient:",
        "Patients   N_sessions",
    ]
)

for n_sessions, n_patients in session_distribution.items():
    report_lines.append(f"{n_patients:<10} {n_sessions}")

# Count unique patients per MLD subtype
subtype_counts = (
    patient_df.groupby(Cols.MLD_FORM)[Cols.SUBJECT_ID]
    .nunique()
    .sort_values(ascending=False)
)

report_lines.extend(["", "Patients per MLD subtype:"])

for subtype, n_patients in subtype_counts.items():
    report_lines.append(f"{str(subtype):<20} {n_patients}")

report_path = Path(__file__).with_suffix(".txt")
report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

# %%
