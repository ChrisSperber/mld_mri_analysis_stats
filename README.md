# A Study on Brain MRI Markers in Metachromatic Leukodystrophy - Statistical Analyses

This repository contains the Python and Bash code, as well as detailed results and documentation, for a research project analysing imaging markers in metachromatic leukodystrophy.

The repository code requires the data master directory "mld_data_master_2026". This data master file follows a BIDS (Brain Imaging Data Structure)-like structure and contains partially confidential data.
> ⚠️ **Note**: This repository does not contain any raw clinical or imaging data.
---

## Repository Contents

| Folder/File                               | Description                                                     |
|-------------------------------------------|-----------------------------------------------------------------|
| `scripts/`                                | Main processing and analysis scripts (Python + Bash)            |
| └──`data_documentation_and_validation/`   | Tracks available data, prerequisite for all other code          |
| └──`anatomical_segmentation/`             | T1 segmentation via synthseg and analysis; also generated T1    |
|                                           |   brain masks and derives the T1-to-FA rigid registration       |
| └──`tract_segmentation_tractseg/`         | Tract segmentation via Tractseg and analysis                    |
| `src/`                                    |                                                                 |
| └──`mld_mri_stats/`                       | Installable Python toolbox (utilities, config paths)            |
| `data/`                                   |                                                                 |
| └──`reference/`                           | Locally stored main files. Partially confidential               |
| └──`snapshot/`                            | Non-confidential documentation of included datasets             |
| `reports/`                                | Environment logs (e.g., Freesurfer, TractSeg, ANTs)             |
| `requirements.txt`                        | Python dependencies (from `venv`)                               |
| `LICENSE`                                 | MIT License                                                     |

---

## Reproducing the Analysis

This project was developed and run using Python 3.12.12 in a local `venv` environment in Ubuntu 22.04.5
All dependency versions outside of Python are tracked in ./reports
> ⚠️ **Note**: Freesurfer 8.1.0 does **not** run on Ubuntu OS versions >22.x

### 1. Clone the repository
```bash
git clone https://github.com/ChrisSperber/mld_mri_analysis_stats
cd mld_mri_analysis_stats
```
### 2. Set up environment
```bash
python -m venv .venv
source .venv/bin/activate
```
### 3. Install dependencies
```bash
pip install -r requirements.txt
# install the local package in editable mode
pip install -e .
```
### 4. Run Analysis

The main analysis scripts are sequentially ordered with alphabetic prefixes. See docstrings for further information.
Scripts in scripts/data_documentation_and_validation are a prerequisite for running all other code in the repository.
Scripts in scripts/anatomical_segmentation were also used to generate brain masks that were required for T1-to-FA rigid registration and, hence, are required to run for all analyses relying on alignment of DWI data with T1 and any other non-DWI data.

---
## Input data
The repository is intended to reside in the same parent directory as the data master directory.

```bash
<parent>/
├─ mld_data_master_2026/     # MRI and derivatives, demographic and clinical tables (not included in repo)
└─ mld_mri_analysis_stats/   # this repository
```

---
## Outputs
### Intermediate Outputs
The code derives various imaging-related data such as segmentation maps and warp fields. These are created in subfolders in "temp_images".
### Outputs
Outputs include volumetric segmentation data in the temp_images folder. All statistical results are stored along with the code. See individual docstrings for more information.

---
### References
TBA

---
### License
This project is licensed under the MIT License — see [Project License](LICENSE) for details.
