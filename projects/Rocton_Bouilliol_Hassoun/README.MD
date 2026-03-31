# 🧬 Leukemia Survival Rate Prediction  
**ENS Data Challenge**

This project focuses on predicting survival outcomes for patients diagnosed with leukemia by combining **clinical hematological data** with **molecular genetic information** (somatic mutations).  
The challenge lies in integrating heterogeneous data sources to build robust, interpretable machine learning models with strong clinical relevance.

---

## 📁 Project Structure

```
Leukemia-Survival-Rate/
├── src/                          # Source code
│   ├── config.py                 # Hyperparameter grids (RSF, XGBoost)
│   ├── DataManagement.py         # Data loading, preprocessing & EDA classes
│   ├── ModelPipelines.py         # sklearn Pipeline builders
│   ├── GridSearch.py             # Model selection & submission generation
│   └── LearningCurve.py          # Learning curve & risk score analysis
│
├── data/
│   ├── raw/                      # Input data (do not modify)
│   │   ├── X_train/
│   │   │   ├── clinical_train.csv
│   │   │   └── molecular_train.csv
│   │   ├── X_test/
│   │   │   ├── clinical_test.csv
│   │   │   └── molecular_test.csv
│   │   └── target_train.csv
│   └── submissions/              # Generated submission files
│       ├── submission_from_gridsearch.csv
│       └── random_submission_FRacdcw_v9kP4pP.csv
│
├── notebooks/                    # Exploratory & benchmark notebooks
│   ├── Sandbox.ipynb
│   └── Benchmark_nqBJ7fO.ipynb
│
├── docs/                         # Reference documents
│   └── Random Number Generation.pdf
│
├── requirements.txt
├── .gitignore
└── README.MD
```

> **Note**: All scripts must be run from the **project root** so that `src.*` imports and `./data/raw/` paths resolve correctly.

---

## 📊 1. Data Understanding

A key difficulty of this dataset is the **structural mismatch between data sources**:

- **Clinical data**:  
  - One row per patient  
  - Represents the global physiological and hematological state

- **Molecular data**:  
  - Multiple rows per patient  
  - One row per detected somatic mutation  

This required careful aggregation and alignment to produce a unified modeling dataset.

---

## 🩺 Clinical Parameters (Patient Health Status)

The clinical variables describe disease severity, bone marrow function, and immune response:

- **BM_BLAST**  
  Percentage of immature blast cells in bone marrow.  
  A primary indicator of disease aggressiveness and progression.

- **WBC / ANC / MONOCYTES**  
  White blood cell counts.  
  Extreme values often reflect immune system saturation or uncontrolled leukemic proliferation.

- **HB (Hemoglobin)**  
  Measures anemia severity, indicating impaired oxygen transport due to marrow dysfunction.

- **PLT (Platelets)**  
  Essential for blood clotting.  
  Low values (thrombocytopenia) suggest bone marrow invasion by malignant cells.

- **CYTOGENETICS**  
  Chromosomal abnormalities derived from karyotype analysis.  
  Particular attention is given to high-risk markers such as:
  - *Complex karyotypes*
  - `+8` (Trisomy 8)
  - `-7` (Monosomy 7)

---

## 🧬 Molecular Parameters (Genetic Alterations)

Molecular features capture the genetic drivers of leukemia:

- **GENE**  
  Identifies the affected gene (e.g., *TP53*, *TET2*, *ASXL1*).

- **VAF (Variant Allele Frequency)**  
  Proportion of cells carrying the mutation.  
  High VAF values suggest clonal dominance.

- **EFFECT**  
  Functional impact of the mutation:
  - `stop_gained` → truncated protein
  - `non_synonymous` → amino acid substitution

---

## 🛠️ 2. Preprocessing & Feature Engineering

To prepare the data for machine learning, several transformations were applied:

### 🔄 Molecular Data Pivoting
- Transformed transactional mutation data into a **one-row-per-patient** format
- Aggregated mutations using the **maximum VAF per gene**, capturing the strongest clonal signal

### 📐 Statistical Management
A custom `DataViewer` class was developed to:

- Analyze variable distributions using **Jarque–Bera normality tests**
- Detect missing values (NaNs) and recommend **mean vs. median imputation**
- Visualize feature correlations to detect and reduce **collinearity**

### 🎯 Target Alignment
- Performed an **inner merge** between `X_train` and `y_train` using patient IDs
- Ensured correct mapping of:
  - `OS_YEARS` (survival duration)
  - `OS_STATUS` (survival outcome)

---

## 📈 3. Exploratory Data Analysis (EDA)

Key insights from exploratory analysis include:

- **Outliers**  
  Standardized boxplots revealed extreme clinical values requiring clipping or special handling.

- **Non-Normal Distributions**  
  Most blood parameters (e.g., WBC, PLT) are highly skewed.  
  Statistical tests suggest applying logarithmic transformations.

- **Survival Patterns**  
  Scatterplots and violin plots were used to visualize relationships between biomarkers and survival outcomes.

---

## 🚀 Quick Start

Run from the **project root**:

```python
# Example: train and generate a submission
python -m src.GridSearch
```

Or import modules directly:

```python
from src.DataManagement import DefaultDataHandler, ImprovedDataHandler
from src.ModelPipelines import XGBoostSurvivalPipeline
from src.GridSearch import ModelSelection
```

---

## 📌 Objective

The ultimate goal is to build a **clinically meaningful and statistically robust survival prediction model**, leveraging both phenotypic and genotypic patient information.

---
