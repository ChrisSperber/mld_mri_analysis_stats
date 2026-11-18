"""Project paths and configuration."""

from pathlib import Path

RNG_SEED = 9001
MAX_WORKERS = 6

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_DIR = REPO_ROOT.parent
TEMPORARY_DATA_DIR = PROJECT_DIR / "temp_images"

DATA_MASTER_DIR = PROJECT_DIR / "mld_data_master_2026"
DATA_DERIVATIVES_DIR = DATA_MASTER_DIR / "derivatives"
DATA_DIFFUSION_DERIVATIVES_DIR = DATA_DERIVATIVES_DIR / "diffusion"
DATA_MTR_DERIVATIVES_DIR = DATA_DERIVATIVES_DIR / "mtr"

SOURCE_MASTER_XLS = DATA_MASTER_DIR / "data_master_2026.xlsx"
MASTER_DATA_CSV = REPO_ROOT / "data" / "reference" / "data_table_master.csv"
MASTER_DATA_CSV_SNAPSHOT = REPO_ROOT / "data" / "snapshot" / "data_table_snapshot.csv"
FIXEL_METRICS_CSV = DATA_DERIVATIVES_DIR / "fixel_analysis" / "tract_metrics.csv"

UNKNOWN = "Unknown"
NOT_APPLICABLE = "Not_applicable"
NOT_AVAILABLE = "Not_available"
