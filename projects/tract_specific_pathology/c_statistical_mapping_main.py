"""Map associations between tract-wise MRI markers and motor impairment in MLD.

This script performs a mass-univariate statistical analysis of the association
between quantitative MRI markers in individual fibre tracts and motor function,
as measured by the Gross Motor Function Classification (GMFC).

Sample
------
Only patient sessions from the project-specific sample reference table are
included. Controls are not considered in this analysis.

Tract-wise MRI measures are obtained from b_extract_tract_measures.csv.
Clinical and demographic variables are obtained from
a_document_sample_reference.csv.

Statistical models
------------------
For each combination of fibre tract and MRI marker, two nested linear
mixed-effects models are fitted:

    Null model: GMFC ~ AgeMRI + (1 | SubjectID)
    Full model: GMFC ~ AgeMRI + Marker + (1 | SubjectID)

Both models include a subject-specific random intercept to account for
the dependence of repeated MRI sessions within subjects.

Baseline covariates can be modified in the configuration section.
Models are fitted using maximum likelihood rather than restricted
maximum likelihood.

Effect size
-----------
The primary effect size is incremental marginal R² (delta R²), calculated
as the difference in Nakagawa-style marginal R² between the full and
null model. It quantifies the incremental explanatory value of the MRI
marker beyond the baseline covariates.

Negative delta R² values are possible because variance components are
re-estimated in each model.

Optionally, confidence intervals are estimated using subject-level
bootstrap resampling. All sessions of a sampled subject are retained.
Repeatedly sampled subjects are assigned independent bootstrap identifiers.
WARNING: Bootstrapping can take several hours!

Statistical inference
---------------------
A likelihood-ratio test compares the full and null model.

Multiple comparisons are corrected using the Holm-Bonferroni procedure,
separately for each MRI marker across all included fibre tracts.
No correction across MRI markers is applied.

Outputs
-------
- CSV table with one row per tract and MRI marker, containing effect sizes,
  confidence intervals, likelihood-ratio test results, regression coefficients,
  and corrected p-values.
- TXT report documenting the sample, model specification, analysis parameters,
  and summary of statistical results.
"""

# %%

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

from mld_mri_stats.config import NOT_AVAILABLE, RNG_SEED
from mld_mri_stats.mixed_models import fit_marker_mixed_model
from mld_mri_stats.utils import Cols, FibreTractMasks

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_PATH = Path(__file__)

REFERENCE_PATH = SCRIPT_PATH.with_name("a_document_sample_reference.csv")
FEATURES_PATH = SCRIPT_PATH.with_name("b_extract_tract_measures.csv")

OUTPUT_PATH = SCRIPT_PATH.with_suffix(".csv")
REPORT_PATH = SCRIPT_PATH.with_name(f"{SCRIPT_PATH.stem}_report.txt")

# Sample and model specification
GROUP = "patient"
TARGET = Cols.GMFC
SUBJECT = Cols.SUBJECT_ID
COVARIATES = ["AgeMRI"]

# MRI markers
MARKERS = [
    "FA",
    "MD",
    "AK",
    "MK",
    "RK",
    "NDI",
    "ODI",
    "FWF",
    "FOD_integral",
    "FD",
    "FC",
    "FDC",
]

# Fibre tracts
TRACTS = [
    FibreTractMasks.CC1,
    FibreTractMasks.CC2,
    FibreTractMasks.CC3,
    FibreTractMasks.CC4,
    FibreTractMasks.CC5,
    FibreTractMasks.CST_L,
    FibreTractMasks.CST_R,
    FibreTractMasks.IFOF_L,
    FibreTractMasks.IFOF_R,
    FibreTractMasks.ILF_L,
    FibreTractMasks.ILF_R,
    FibreTractMasks.SLF1_L,
    FibreTractMasks.SLF1_R,
    FibreTractMasks.SLF2_L,
    FibreTractMasks.SLF2_R,
    FibreTractMasks.SLF3_L,
    FibreTractMasks.SLF3_R,
]

# Bootstrap configuration
N_BOOTSTRAP = 500  # Set to 0 to disable bootstrapping
CI_LEVEL = 0.95
RANDOM_STATE = RNG_SEED

# Statistical inference
ALPHA = 0.05
CORRECTION_METHOD = "holm"

# Table identifiers
SESSION_KEYS = [Cols.SUBJECT_ID, Cols.SESSION_ID]


# %%
# =============================================================================
# Load and prepare data
# =============================================================================

reference = pd.read_csv(
    REFERENCE_PATH,
    sep=";",
    na_values=[NOT_AVAILABLE],
    dtype={
        Cols.SUBJECT_ID: str,
        Cols.SESSION_ID: str,
    },
)

features = pd.read_csv(
    FEATURES_PATH,
    sep=";",
    na_values=[NOT_AVAILABLE],
    dtype={
        Cols.SUBJECT_ID: str,
        Cols.SESSION_ID: str,
    },
)

# Select relevant sample
reference = reference[reference[Cols.GROUP] == GROUP].copy()

# Calculate age at MRI in years
for column in [Cols.DOB, Cols.DATE_MRI]:
    reference[column] = pd.to_datetime(
        reference[column].astype(str),
        format="%Y%m%d",
        errors="raise",
    )

reference["AgeMRI"] = (reference[Cols.DATE_MRI] - reference[Cols.DOB]).dt.days / 365.25

# Select relevant clinical variables
clinical_columns = SESSION_KEYS + [TARGET] + COVARIATES
clinical = reference[clinical_columns].copy()

# Select relevant imaging features
features = features[features["Tract"].isin(TRACTS)].copy()

# Merge imaging features with clinical data
df = features.merge(
    clinical,
    on=SESSION_KEYS,
    how="inner",
    validate="many_to_one",
)

# Convert outcome and MRI markers to numeric values
for column in [TARGET] + MARKERS:
    df[column] = pd.to_numeric(df[column], errors="raise")


# %%
# =============================================================================
# Validate analysis data
# =============================================================================

# Check completeness of subject and session identifiers
if reference[SESSION_KEYS].isna().any().any():
    raise ValueError("Missing subject or session identifiers in reference table.")

# Check completeness of clinical and imaging data
required_columns = SESSION_KEYS + ["Tract", TARGET] + COVARIATES + MARKERS
missing = df[required_columns].isna().sum()

if missing.any():
    raise ValueError(f"Missing values in analysis data:\n{missing[missing > 0]}")

# Check for non-finite numeric values
numeric_columns = (
    [TARGET]
    + MARKERS
    + [column for column in COVARIATES if pd.api.types.is_numeric_dtype(df[column])]
)

if not np.isfinite(df[numeric_columns].to_numpy(dtype=float)).all():
    raise ValueError("Non-finite numeric values in analysis data.")

# Check uniqueness of session-tract combinations
if df.duplicated(SESSION_KEYS + ["Tract"]).any():
    raise ValueError("Duplicate session-tract combinations in analysis data.")

# Verify that every reference session has data for every included tract
expected_sessions = pd.MultiIndex.from_frame(reference[SESSION_KEYS])

for tract in TRACTS:
    tract_sessions = pd.MultiIndex.from_frame(
        df.loc[df["Tract"] == tract, SESSION_KEYS]
    )

    missing_sessions = expected_sessions.difference(tract_sessions)

    if len(missing_sessions) > 0:
        raise ValueError(f"Missing sessions for tract {tract}:\n{missing_sessions}")

# All tract-marker models should now use identical samples
n_subjects = reference[SUBJECT].nunique()
n_sessions = len(reference)
n_models = len(TRACTS) * len(MARKERS)

print(f"Subjects: {n_subjects}")
print(f"Sessions: {n_sessions}")
print(f"Tracts: {len(TRACTS)}")
print(f"MRI markers: {len(MARKERS)}")
print(f"Model comparisons: {n_models}")
print(f"Bootstrap iterations per comparison: {N_BOOTSTRAP}")


# %%
# =============================================================================
# Statistical mapping
# =============================================================================

results = []

with tqdm(total=n_models, desc="Statistical mapping") as progress:

    for tract in TRACTS:
        tract_data = df[df["Tract"] == tract].copy()

        for marker in MARKERS:
            progress.set_postfix(tract=tract, marker=marker)

            result = fit_marker_mixed_model(
                df=tract_data,
                marker_colname=marker,
                target_var_colname=TARGET,
                subject_colname=SUBJECT,
                covariate_colnames=COVARIATES,
                n_bootstrap=N_BOOTSTRAP,
                ci_level=CI_LEVEL,
                random_state=RANDOM_STATE,
            )

            results.append(
                {
                    "Tract": tract,
                    "Marker": marker,
                    **asdict(result),
                }
            )

            progress.update(1)

results_df = pd.DataFrame(results)


# %%
# =============================================================================
# Multiple-comparison correction
# =============================================================================

# Correct across tracts separately for each MRI marker
results_df["lrt_pvalue_corrected"] = np.nan

for marker in MARKERS:
    marker_mask = results_df["Marker"] == marker
    pvalues = results_df.loc[marker_mask, "lrt_pvalue"]

    _, pvalues_corrected, _, _ = multipletests(
        pvalues,
        alpha=ALPHA,
        method=CORRECTION_METHOD,
    )

    results_df.loc[marker_mask, "lrt_pvalue_corrected"] = pvalues_corrected


# %%
# =============================================================================
# Save statistical results
# =============================================================================

results_df.to_csv(
    OUTPUT_PATH,
    sep=";",
    index=False,
    na_rep=NOT_AVAILABLE,
)


# %%
# =============================================================================
# Generate analysis report
# =============================================================================

baseline_terms = " + ".join(COVARIATES) or "1"

formula_null = f"{TARGET} ~ {baseline_terms} + (1 | {SUBJECT})"
formula_full = f"{TARGET} ~ {baseline_terms} + Marker + (1 | {SUBJECT})"

# Summarise statistical results per MRI marker
summary = []

for marker in MARKERS:
    marker_results = results_df[results_df["Marker"] == marker]

    summary.append(
        {
            "Marker": marker,
            "Median delta R2": marker_results["r2_delta"].median(),
            "Maximum delta R2": marker_results["r2_delta"].max(),
            "Significant uncorrected": (marker_results["lrt_pvalue"] < ALPHA).sum(),
            "Significant corrected": (
                marker_results["lrt_pvalue_corrected"] < ALPHA
            ).sum(),
        }
    )

summary_df = pd.DataFrame(summary).set_index("Marker")

with REPORT_PATH.open("w", encoding="utf-8") as file:
    file.write("TRACT-WISE STATISTICAL MAPPING\n")
    file.write("==============================\n\n")

    file.write("DATA SOURCES\n")
    file.write("------------\n")
    file.write(f"Sample reference: {REFERENCE_PATH.name}\n")
    file.write(f"Imaging features: {FEATURES_PATH.name}\n\n")

    file.write("SAMPLE\n")
    file.write("------\n")
    file.write(f"Group: {GROUP}\n")
    file.write(f"N subjects: {n_subjects}\n")
    file.write(f"N sessions: {n_sessions}\n")
    file.write(f"N tracts: {len(TRACTS)}\n")
    file.write(f"N MRI markers: {len(MARKERS)}\n")
    file.write(f"N model comparisons: {len(results_df)}\n\n")

    file.write("MODEL SPECIFICATION\n")
    file.write("-------------------\n")
    file.write(f"Null model: {formula_null}\n")
    file.write(f"Full model: {formula_full}\n")
    file.write("Estimation: maximum likelihood\n")
    file.write("Effect size: incremental marginal R2\n")
    file.write("Statistical inference: likelihood-ratio test\n\n")

    file.write("MULTIPLE-COMPARISON CORRECTION\n")
    file.write("------------------------------\n")
    file.write(f"Correction method: {CORRECTION_METHOD}\n")
    file.write("Correction family: tracts within each MRI marker\n")
    file.write(f"Tests per family: {len(TRACTS)}\n")
    file.write(f"Alpha: {ALPHA}\n\n")

    file.write("BOOTSTRAPPING\n")
    file.write("-------------\n")
    file.write(f"Iterations per model comparison: {N_BOOTSTRAP}\n")
    file.write(f"Random seed: {RANDOM_STATE}\n")

    if N_BOOTSTRAP > 0:
        file.write("Resampling unit: subject\n")
        file.write("CI method: percentile bootstrap\n")
        file.write(f"Confidence level: {CI_LEVEL}\n")
        file.write(
            "Minimum valid bootstrap iterations: "
            f"{results_df['n_bootstrap_valid'].min()}\n"
        )
        file.write(
            "Median valid bootstrap iterations: "
            f"{results_df['n_bootstrap_valid'].median():.1f}\n"
        )
        file.write(
            "Maximum valid bootstrap iterations: "
            f"{results_df['n_bootstrap_valid'].max()}\n"
        )

    file.write("\n")

    file.write("INCLUDED TRACTS\n")
    file.write("---------------\n")
    file.write(", ".join(TRACTS) + "\n\n")

    file.write("INCLUDED MRI MARKERS\n")
    file.write("--------------------\n")
    file.write(", ".join(MARKERS) + "\n\n")

    file.write("SUMMARY OF STATISTICAL RESULTS\n")
    file.write("------------------------------\n")
    file.write(summary_df.to_string(float_format=lambda x: f"{x:.4f}"))
    file.write("\n")

print(f"Statistical results saved to: {OUTPUT_PATH}")
print(f"Analysis report saved to: {REPORT_PATH}")

# %%
