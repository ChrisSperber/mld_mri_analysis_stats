"""Document the sample included in the tract-specific pathology analysis.

Includes all controls and patient sessions with successful FOD registration.

Categorical characteristics are reported at subject level, based on the
first included session. Age and clinical scores are reported at session level.

Outputs:
- CSV snapshot of selected sessions and clinical characteristics.
- CSV reference table containing all master table columns for selected sessions.
- Markdown document containing descriptive cohort statistics.
"""

# %%

from pathlib import Path

import pandas as pd

from mld_mri_stats.config import MASTER_DATA_CSV, NOT_AVAILABLE
from mld_mri_stats.utils import SNAPSHOT_COLS, Cols, PathCols, describe_cohort

SCRIPT_PATH = Path(__file__)
SNAPSHOT_PATH = SCRIPT_PATH.with_name(f"{SCRIPT_PATH.stem}_snapshot.csv")
REFERENCE_PATH = SCRIPT_PATH.with_name(f"{SCRIPT_PATH.stem}_reference.csv")
MD_PATH = SCRIPT_PATH.with_suffix(".md")


# %%
# Load data
df = pd.read_csv(MASTER_DATA_CSV, na_values=[NOT_AVAILABLE], sep=";")

# Select relevant sessions
is_patient_fod = (df[Cols.GROUP] == "patient") & df[PathCols.TRACT_CC1].notna()
is_control = df[Cols.GROUP] == "control"

df = df.loc[is_patient_fod | is_control].copy()


# Save sample snapshot
df[SNAPSHOT_COLS].to_csv(
    SNAPSHOT_PATH,
    sep=";",
    index=False,
    na_rep=NOT_AVAILABLE,
)

# Save full sample reference table
df.to_csv(
    REFERENCE_PATH,
    sep=";",
    index=False,
    na_rep=NOT_AVAILABLE,
)


# Convert dates and calculate age at MRI
for column in [Cols.DOB, Cols.DATE_MRI]:
    df[column] = pd.to_datetime(
        df[column].astype("Int64").astype(str),
        format="%Y%m%d",
        errors="coerce",
    )

df["AgeMRI"] = (df[Cols.DATE_MRI] - df[Cols.DOB]).dt.days / 365.25

# Convert clinical scores to numeric values
for column in [Cols.GMFC, Cols.MLD_MRI_SCORE]:
    df[column] = pd.to_numeric(df[column], errors="coerce")


# Define cohorts
patients_fod = df[df[Cols.GROUP] == "patient"]
controls = df[df[Cols.GROUP] == "control"]


# %%
# Generate descriptive statistics
table = pd.DataFrame(
    {
        "Patients (FOD)": describe_cohort(patients_fod),
        "Controls": describe_cohort(controls),
    }
)

table.index.name = "Characteristic"

with MD_PATH.open("w", encoding="utf-8") as file:
    file.write("# Descriptive cohort statistics\n\n")
    file.write(
        "Sample: all controls and patient sessions "
        "with successful FOD registration.\n\n"
    )
    file.write(table.to_markdown())
    file.write(
        "\n\n"
        "Sessions per subject: median (range).\n\n"
        "Age and clinical scores: median [Q1–Q3]; (range).\n\n"
        "Categorical variables: counts of unique subjects, "
        "based on their first included session.\n\n"
        "Missing values are omitted from descriptive statistics.\n"
    )

print(f"Sample snapshot saved to: {SNAPSHOT_PATH}")
print(f"Sample reference saved to: {REFERENCE_PATH}")
print(f"Descriptive statistics saved to: {MD_PATH}")

# %%
