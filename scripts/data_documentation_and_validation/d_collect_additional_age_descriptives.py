"""Generate age–GMFC scatterplots stratified by MLD type.

Total patient sample and subsample with successful FOD registration
are plotted separately. Each observation represents one MRI session.

Output:
- PNG with age–GMFC scatterplots
"""

# %%

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mld_mri_stats.config import MASTER_DATA_CSV, NOT_AVAILABLE
from mld_mri_stats.utils import Cols, PathCols

SCRIPT_PATH = Path(__file__)
FIGURE_PATH = SCRIPT_PATH.with_name(f"{SCRIPT_PATH.stem}_age_gmfc_scatter.png")


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

# Convert GMFC to numeric values
df[Cols.GMFC] = pd.to_numeric(df[Cols.GMFC], errors="coerce")


# Define cohorts
patients = df[df[Cols.GROUP] == "patient"]
patients_fod = patients[patients[PathCols.TRACT_CC1].notna()]


# %%
# Define consistent colours for MLD types
mld_types = sorted(patients[Cols.MLD_FORM].dropna().unique())

colours = {mld_type: plt.get_cmap("tab10")(i) for i, mld_type in enumerate(mld_types)}


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
    plot_data = cohort[["AgeMRI", Cols.GMFC, Cols.MLD_FORM]].dropna()

    for mld_type in mld_types:
        type_data = plot_data[plot_data[Cols.MLD_FORM] == mld_type]

        x = type_data["AgeMRI"].to_numpy()
        y = type_data[Cols.GMFC].to_numpy()

        # Add slight jitter to overlapping GMFC observations
        y_jittered = y + rng.uniform(-0.12, 0.12, size=len(y))

        ax.scatter(
            x,
            y_jittered,
            color=colours[mld_type],
            label=mld_type,
            alpha=0.65,
            s=30,
        )

    ax.set_title(f"{title}\n(n = {len(plot_data)} sessions)")
    ax.set_xlabel("Age at MRI (years)")
    ax.set_yticks(range(7))
    ax.set_ylim(-0.5, 6.5)
    ax.grid(alpha=0.2)
    ax.legend(title="MLD type")

axes[0].set_ylabel("GMFC")

fig.tight_layout()
fig.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Figure saved to: {FIGURE_PATH}")

# %%
