"""Generate descriptive cohort statistics and GMFC–MLD MRI score scatterplots.

Total patient sample and subsample with succesful FOD registration are separately reported.
MLD Type, Therapy Type, and Sex are reported on subject level; GMFC and MLD MRI score on session
level.

Outputs:
- Markdown document containing human-readable information on samples
- png with scatterplot GMFC against MLD MRI Score

"""

# %%

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mld_mri_stats.config import MASTER_DATA_CSV, NOT_AVAILABLE
from mld_mri_stats.utils import Cols, PathCols

SCRIPT_PATH = Path(__file__)
TABLE_PATH = SCRIPT_PATH.with_name(f"{SCRIPT_PATH.stem}_tables.md")
FIGURE_PATH = SCRIPT_PATH.with_name(f"{SCRIPT_PATH.stem}_gmfc_mri_scatter.png")


def median_range(series):
    """Format median and range."""
    series = series.dropna()

    if series.empty:
        return "–"

    return f"{series.median():.1f} " f"({series.min():.1f}–{series.max():.1f})"


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
    """Calculate descriptive statistics for a cohort."""
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


# %%
# Load data
df = pd.read_csv(MASTER_DATA_CSV, na_values=[NOT_AVAILABLE], sep=";")

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
patients = df[df[Cols.GROUP] == "patient"]
patients_fod = patients[patients[PathCols.TRACT_CC1].notna()]
controls = df[df[Cols.GROUP] == "control"]


# Generate descriptive statistics
table = pd.DataFrame(
    {
        "All patients": describe_cohort(patients),
        "Patients (FOD)": describe_cohort(patients_fod),
        "Controls": describe_cohort(controls),
    }
)

table.index.name = "Characteristic"

with TABLE_PATH.open("w", encoding="utf-8") as file:
    file.write("# Descriptive cohort statistics\n\n")
    file.write(table.to_markdown())
    file.write(
        "\n\n"
        "Sessions per subject: median (range).\n\n"
        "Age and clinical scores: median [Q1–Q3]; (range).\n\n"
        "Categorical variables: counts of unique subjects, "
        "based on their first included session.\n\n"
        "Missing values are omitted from descriptive statistics.\n"
    )


# Generate scatterplots
rng = np.random.default_rng(42)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(10, 4.5),
    sharex=True,
    sharey=True,
)

for ax, cohort, title in zip(
    axes,
    [patients, patients_fod],
    ["All patients", "Patients with FOD registration"],
    strict=True,
):
    plot_data = cohort[[Cols.MLD_MRI_SCORE, Cols.GMFC]].dropna()

    x = plot_data[Cols.MLD_MRI_SCORE].to_numpy()
    y = plot_data[Cols.GMFC].to_numpy()

    # Add slight jitter to overlapping observations
    x_jittered = x + rng.uniform(-0.15, 0.15, size=len(x))
    y_jittered = y + rng.uniform(-0.12, 0.12, size=len(y))

    ax.scatter(x_jittered, y_jittered, alpha=0.65, s=30)

    ax.set_title(f"{title}\n(n = {len(plot_data)} sessions)")
    ax.set_xlabel("MLD MRI score")
    ax.set_yticks(range(7))
    ax.set_ylim(-0.5, 6.5)
    ax.grid(alpha=0.2)

axes[0].set_ylabel("GMFC")

fig.tight_layout()
fig.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Table saved to: {TABLE_PATH}")
print(f"Figure saved to: {FIGURE_PATH}")

# %%
