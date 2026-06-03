# Wisconsin Circuit Court Recidivism Prediction

An open-source, socio-economically aware recidivism prediction model trained
on the Wisconsin Circuit Court Longitudinal Data (WCLD) dataset. The project
quantifies how neighborhood-level census features shape 180-day recidivism
risk, and exports interactive dashboard
artefacts.


Interactive Dashboard Link: https://recidivismpredictionwcc-kfyajwz7quye87vxkogsah.streamlit.app/
Model Ready Data Google Drive Link: https://drive.google.com/drive/folders/12J2RHq5HRX9JDTeJUPCvzlfXN3xOyjSV?usp=drive_link

---


## Project Overview

The dominant commercial recidivism tool used in U.S. courtrooms, COMPAS, is
proprietary, unauditable, and has been shown to produce racially disparate
predictions. This project presents a transparent alternative that:

- Matches COMPAS predictive performance (AUC-ROC 0.7043 vs. published 0.65–0.70)
- Explicitly quantifies how neighborhood socioeconomic context shapes risk
- Introduces a Socio-Economic Deprivation Index (SEDI) engineered from
  census tract data
- Shows a fairness audit: TPR, FPR, demographic parity by race
- Exports dashboard files for non-technical stakeholder exploration

All code, trained models, SHAP artefacts, and dashboard exports are released
publicly so that the community can inspect,
reproduce, and improve every component of the system.


---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/studnt001/Recidivism_Prediction_WCC.git
cd wi-recidivism

# 2. Optional, create and activate a virtual environment 
python -m venv venv
source venv/bin/activate          # macOS / Linux
venv\Scripts\activate             # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the raw data (see Data section below)
#    Place wcld.csv in data/raw/

# 5. Run the EDA notebook
jupyter notebook notebooks/01_EDA_and_Cleaning.ipynb

# 6. Run the modeling notebook
jupyter notebook notebooks/02_Neighborhood_Model.ipynb
```

The modeling notebook auto-installs `xgboost`, `shap`, `imbalanced-learn`,
and `plotly` if they are not already present. All other dependencies are
covered by `requirements.txt`.

---

## Requirements

### Python version

Python 3.8 or higher is required. The project was developed and tested on
Python 3.10.

### Install all dependencies

```bash
pip install -r requirements_1.txt
```

### requirements_1.txt

```
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
xgboost>=1.7.0
shap>=0.41.0
imbalanced-learn>=0.10.0
plotly>=5.13.0
scipy>=1.9.0
ipykernel>=6.0.0
jupyter>=1.0.0
```


---

## Data

### Source

The Wisconsin Circuit Court Longitudinal Data (WCLD) is compiled by Ash,
Goel, Li, Marangon, and Sun at ETH Zurich and the University of Oxford.

- **Download:** https://clezdata.github.io/wcld/
- **File:** `wcld.csv`
- **Rows:** 1,476,967
- **Columns:** 54
- **Coverage:** Wisconsin circuit court cases filed 2000–2018

### Placement

After downloading, place the raw file at:

```
data/raw/wcld.csv
```

Then open `notebooks/01_EDA_and_Cleaning.ipynb` and update the `DATA_PATH`
variable in the first code cell to match your local path:

```python
DATA_PATH = 'data/raw/wcld.csv'
```

### What the EDA notebook produces

Running `01_EDA_and_Cleaning.ipynb` end to end writes two CSV files:

| File | Description |
|------|-------------|
| `data/wcld_clean.csv` | All 1,476,957 rows after cleaning, 63 columns |
| `data/wcld_model_ready.csv` | 1,357,746 model-ready rows with `recid_180d` observed |

The modeling notebook (`02_Neighborhood_Model.ipynb`) reads
`wcld_model_ready.csv`. Update `DATA_PATH` in its first code cell to point to
this file:

```python
DATA_PATH = 'data/wcld_model_ready.csv'
```

### Target variable

`recid_180d` is a binary indicator equal to 1 if the defendant was charged
with a new offense within 180 days of the case disposition date.

- **Observed rows:** 1,357,746 (91.9% of total)
- **Recidivism rate:** 42.21%

---


## Citation


The underlying dataset:

```bibtex
@article{ash2023wcld,
      title={WCLD: Curated Large Dataset of Criminal Cases from Wisconsin Circuit Courts}, 
      author={Elliott Ash and Naman Goel and Nianyun Li and Claudia Marangon and Peiyao Sun},
      booktitle={37th Conference on Neural Information Processing Systems (NeurIPS 2023) Track on Datasets and Benchmarks.},
      year={2023}
}
```

---

## License


The WCLD dataset is subject to its own terms of use available at
https://clezdata.github.io/wcld/. Please review those terms before
redistributing the data.


