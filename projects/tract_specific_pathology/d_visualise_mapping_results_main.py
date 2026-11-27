"""Create a heatmap and supplementary table from tract-wise statistical results.

This script visualises the tract-by-marker statistical mapping results from
c_statistical_mapping.py.

It produces two types of output:

1) Heatmap
   - Rows: MRI markers
   - Columns: fibre tracts
   - Colour: incremental marginal R² (delta R²)
   - Negative delta R² values are clipped to 0 for visualisation only
   - Significant cells after Holm correction are marked with a uniform dot

2) Supplementary table
   - Rows: fibre tracts; columns: MRI markers
   - Each cell contains:
         "delta R² (CI lower, CI upper); p=corrected p; β=marker coefficient"
   - This is intended as a detailed manuscript/supplementary table

The script performs no new statistical analysis. It only reads the existing
results table, reshapes it, formats it, and writes visual outputs.

Outputs
-------
- d_visualise_results_heatmap.png
- d_visualise_results_table.csv
- d_visualise_results_table.md
"""

# %%

from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mld_mri_stats.config import NOT_AVAILABLE
from mld_mri_stats.utils import FibreTractMasks

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_PATH = Path(__file__)
RESULTS_PATH = SCRIPT_PATH.with_name("c_statistical_mapping_main.csv")

HEATMAP_PNG_PATH = SCRIPT_PATH.with_name(f"{SCRIPT_PATH.stem}_heatmap.png")
TABLE_CSV_PATH = SCRIPT_PATH.with_name(f"{SCRIPT_PATH.stem}_table.csv")
TABLE_MD_PATH = SCRIPT_PATH.with_name(f"{SCRIPT_PATH.stem}_table.md")

# Tract order (heatmap columns, table rows)
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

# Marker order (heatmap rows, table columns)
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

# Display labels
TRACT_LABELS = {
    FibreTractMasks.CC1: "CCI",
    FibreTractMasks.CC2: "CCII",
    FibreTractMasks.CC3: "CCIII",
    FibreTractMasks.CC4: "CCIV",
    FibreTractMasks.CC5: "CCV",
    FibreTractMasks.CST_L: "CST L",
    FibreTractMasks.CST_R: "CST R",
    FibreTractMasks.IFOF_L: "IFOF L",
    FibreTractMasks.IFOF_R: "IFOF R",
    FibreTractMasks.ILF_L: "ILF L",
    FibreTractMasks.ILF_R: "ILF R",
    FibreTractMasks.SLF1_L: "SLF I L",
    FibreTractMasks.SLF1_R: "SLF I R",
    FibreTractMasks.SLF2_L: "SLF II L",
    FibreTractMasks.SLF2_R: "SLF II R",
    FibreTractMasks.SLF3_L: "SLF III L",
    FibreTractMasks.SLF3_R: "SLF III R",
}

MARKER_LABELS = {
    "FA": "FA",
    "MD": "MD",
    "AK": "AK",
    "MK": "MK",
    "RK": "RK",
    "NDI": "NDI",
    "ODI": "ODI",
    "FWF": "FWF",
    "FOD_integral": "FOD",
    "FD": "FD",
    "FC": "FC",
    "FDC": "FDC",
}

# Group separators for easier reading
TRACT_GROUP_BREAKS = [5, 7, 9, 11]
MARKER_GROUP_BREAKS = [2, 5, 8, 9]

# Heatmap style
COLORMAP = "YlOrRd"
MISSING_COLOR = "#d9d9d9"
SIGNIFICANCE_ALPHA = 0.05
SIGNIFICANCE_SYMBOL = "●"
SIGNIFICANCE_FONT_SIZE = 10
FIGSIZE = (12, 7)
DPI = 300
FONT_SIZE = 10
TITLE = "Tract-wise incremental marginal R²"


# Table formatting
R2_DECIMALS = 3
P_DECIMALS = 3
BETA_DECIMALS = 3
P_SMALL_THRESHOLD = 0.001
MISSING_TEXT = "–"

# Expected columns in the results file
REQUIRED_COLUMNS = [
    "Tract",
    "Marker",
    "r2_delta",
    "lrt_pvalue_corrected",
    "beta_marker",
]

OPTIONAL_CI_COLUMNS = [
    "r2_delta_ci_low",
    "r2_delta_ci_high",
]

# %%
# =============================================================================
# Helpers
# =============================================================================


def format_pvalue(pvalue: float) -> str:
    """Format p-values for manuscript-style reporting."""
    if pd.isna(pvalue):
        return MISSING_TEXT

    if pvalue < P_SMALL_THRESHOLD:
        return "<0.001"

    return f"{pvalue:.{P_DECIMALS}f}"


def format_r2(value: float) -> str:
    """Format delta R² values."""
    if pd.isna(value):
        return MISSING_TEXT

    return f"{value:.{R2_DECIMALS}f}"


def format_ci(low: float, high: float) -> str:
    """Format confidence intervals."""
    if pd.isna(low) or pd.isna(high):
        return f"({MISSING_TEXT}, {MISSING_TEXT})"

    return f"({low:.{R2_DECIMALS}f}, {high:.{R2_DECIMALS}f})"


def format_table_cell(row: pd.Series) -> str:
    """Create manuscript-style cell text for one tract-marker result."""
    r2_text = format_r2(row["r2_delta"])
    ci_text = format_ci(
        row.get("r2_delta_ci_low", np.nan), row.get("r2_delta_ci_high", np.nan)
    )
    p_text = format_pvalue(row["lrt_pvalue_corrected"])

    beta = row["beta_marker"]
    beta_text = MISSING_TEXT if pd.isna(beta) else f"{beta:.{BETA_DECIMALS}f}"
    p_label = f"p{p_text}" if p_text.startswith("<") else f"p={p_text}"

    return f"{r2_text} {ci_text}; {p_label}; β={beta_text}"


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Convert a DataFrame to a simple markdown table without extra dependencies."""
    headers = [df.index.name or ""] + list(df.columns)

    lines = []
    lines.append("| " + " | ".join(headers) + " |")  # type: ignore
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for index_value, row in df.iterrows():
        values = [str(index_value)] + [str(value) for value in row.tolist()]
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


# %%
# =============================================================================
# Load and validate results
# =============================================================================

results = pd.read_csv(
    RESULTS_PATH,
    sep=";",
    na_values=[NOT_AVAILABLE],
)

missing_required = [
    column for column in REQUIRED_COLUMNS if column not in results.columns
]
if missing_required:
    raise ValueError(f"Missing required columns in results file: {missing_required}")

for column in OPTIONAL_CI_COLUMNS:
    if column not in results.columns:
        results[column] = np.nan

# Restrict to expected tracts and markers
results = results[
    results["Tract"].isin(TRACTS) & results["Marker"].isin(MARKERS)
].copy()

# Check uniqueness of tract-marker combinations
if results.duplicated(["Tract", "Marker"]).any():
    duplicates = results.loc[results.duplicated(["Tract", "Marker"], keep=False)]
    raise ValueError(
        "Duplicate tract-marker combinations in results file:\n"
        f"{duplicates[['Tract', 'Marker']].to_string(index=False)}"
    )

# Check completeness
expected = pd.MultiIndex.from_product([TRACTS, MARKERS], names=["Tract", "Marker"])
observed = pd.MultiIndex.from_frame(results[["Tract", "Marker"]])

missing_pairs = expected.difference(observed)
if len(missing_pairs) > 0:
    raise ValueError(f"Missing tract-marker combinations:\n{missing_pairs}")

# %%
# =============================================================================
# Prepare heatmap matrices
# =============================================================================

delta_r2 = results.pivot(index="Tract", columns="Marker", values="r2_delta").reindex(
    index=TRACTS, columns=MARKERS
)

pvalue_corrected = results.pivot(
    index="Tract", columns="Marker", values="lrt_pvalue_corrected"
).reindex(index=TRACTS, columns=MARKERS)

# Clip negative values to zero for visualisation only
delta_r2_plot = delta_r2.clip(lower=0)

# Significance mask
significant = pvalue_corrected < SIGNIFICANCE_ALPHA

# Colour scale
vmin = 0.0
vmax = float(np.nanmax(delta_r2_plot.to_numpy()))

if not np.isfinite(vmax) or vmax <= 0:
    vmax = 1.0

# %%
# =============================================================================
# Create heatmap
# =============================================================================

plt.rcParams.update({"font.size": FONT_SIZE})

fig, ax = plt.subplots(figsize=FIGSIZE)

cmap = plt.get_cmap(COLORMAP).copy()
cmap.set_bad(MISSING_COLOR)

image = ax.imshow(
    np.ma.masked_invalid(delta_r2_plot.to_numpy().T),
    cmap=cmap,
    vmin=vmin,
    vmax=vmax,
    aspect="auto",
)

# Axis labels
ax.set_xticks(np.arange(len(TRACTS)))
ax.set_xticklabels([TRACT_LABELS[tract] for tract in TRACTS], rotation=45, ha="right")

ax.set_yticks(np.arange(len(MARKERS)))
ax.set_yticklabels([MARKER_LABELS[marker] for marker in MARKERS])

ax.set_xlabel("Fibre tract")
ax.set_ylabel("MRI marker")
ax.set_title(TITLE)

# Add significance dots
for row_idx, marker in enumerate(MARKERS):
    for col_idx, tract in enumerate(TRACTS):
        if bool(significant.loc[tract, marker]):
            ax.text(
                col_idx,
                row_idx,
                SIGNIFICANCE_SYMBOL,
                ha="center",
                va="center",
                color="black",
                fontsize=SIGNIFICANCE_FONT_SIZE,
                path_effects=[
                    pe.withStroke(linewidth=1.5, foreground="white"),
                ],
            )

# Group separators
for break_idx in TRACT_GROUP_BREAKS:
    ax.axvline(break_idx - 0.5, color="white", linewidth=5)

for break_idx in MARKER_GROUP_BREAKS:
    ax.axhline(break_idx - 0.5, color="white", linewidth=5)

# Thin cell borders
ax.set_xticks(np.arange(-0.5, len(TRACTS), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(MARKERS), 1), minor=True)
ax.grid(which="minor", color="white", linestyle="-", linewidth=0.5)
ax.tick_params(which="minor", bottom=False, left=False)

# Colorbar
cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Incremental marginal R² (ΔR²)")

fig.tight_layout()
fig.savefig(HEATMAP_PNG_PATH, dpi=DPI, bbox_inches="tight")
plt.close(fig)

# %%
# =============================================================================
# Create supplementary table
# =============================================================================

results["CellText"] = results.apply(format_table_cell, axis=1)

table_df = results.pivot(index="Tract", columns="Marker", values="CellText").reindex(
    index=TRACTS, columns=MARKERS
)

table_df.index = [TRACT_LABELS[tract] for tract in table_df.index]
table_df.columns = [MARKER_LABELS[marker] for marker in table_df.columns]
table_df.index.name = "Tract"

table_df.to_csv(
    TABLE_CSV_PATH,
    sep=";",
    index=True,
    na_rep=MISSING_TEXT,
)

with TABLE_MD_PATH.open("w", encoding="utf-8") as file:
    file.write("# Supplementary tract-wise statistical results\n\n")
    file.write(
        "Each cell contains: "
        '"delta R² (CI lower, CI upper); p=corrected p; β=marker coefficient".\n\n'
    )
    file.write(
        "Corrected p-values are the Holm-adjusted likelihood-ratio-test p-values "
        "within each marker across tracts.\n\n"
    )
    file.write(
        "Negative delta R² values are retained in this table. "
        "Only the heatmap clips negative values to zero for visualisation.\n\n"
    )
    file.write(dataframe_to_markdown(table_df))
    file.write("\n")

# %%
# =============================================================================
# Done
# =============================================================================

print(f"Heatmap saved to: {HEATMAP_PNG_PATH}")
print(f"Supplementary table saved to: {TABLE_CSV_PATH}")
print(f"Supplementary table saved to: {TABLE_MD_PATH}")

# %%
