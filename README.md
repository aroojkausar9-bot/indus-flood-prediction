# Indus Basin Flood Prediction using Machine Learning

A sub-basin-scale monthly flood occurrence prediction framework for the Indus River Basin in Pakistan, using satellite-derived hydro-meteorological data and machine learning models with strict temporal and cross-basin validation.

## 📄 Paper

> **"Flood Prediction in the Indus Basin in Pakistan Using Supervised Machine Learning Models With Cross-Basin Validation"**  
> Arooj Kausar, SEECS, NUST, Islamabad, Pakistan

---

## 🗂️ Project Structure

```
indus-flood-prediction/
├── data/                          # Raw and processed data (not tracked by git)
│   └── .gitkeep
├── notebooks/
│   ├── 01_data_collection.ipynb   # GEE data extraction (CHIRPS, TerraClimate, SRTM)
│   ├── 02_preprocessing.ipynb     # Data cleaning, merging, flood labeling
│   ├── 03_feature_engineering.ipynb # Temporal + spatial feature creation
│   └── 04_modeling_results.ipynb  # RF, SVM-PSO training, evaluation, visualization
├── src/
│   ├── data_collection.py         # GEE extraction scripts
│   ├── preprocessing.py           # Preprocessing pipeline
│   ├── feature_engineering.py     # Temporal & spatial features
│   ├── models.py                  # RF and SVM-PSO model classes
│   ├── spatial_features.py        # Coordinate extraction and spatial feature builder
│   └── evaluation.py              # Metrics, confusion matrix, ROC utilities
├── models/                        # Saved model artifacts (not tracked by git)
│   └── .gitkeep
├── results/                       # Output CSVs and figures
│   └── .gitkeep
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🌍 Study Area

- **34 sub-basins** of the Indus River Basin within Pakistan
- Sub-basin boundaries from **HydroBASINS** (Level 5), clipped to Pakistan using Natural Earth polygons
- Time period: **2010–2024** (monthly resolution)

---

## 📦 Data Sources

| Dataset | Source | Variable |
|---|---|---|
| Rainfall | CHIRPS Daily (GEE) | `rainfall_mm` |
| Temperature | TerraClimate (GEE) | `tmax_value` |
| Topography | SRTM 30m (GEE) | `elev_mean`, `slope_mean` |
| Flood Labels (2010–2018) | Global Flood Database (GEE) | `Flood_Flag` |
| Flood Labels (2019–2024) | Rainfall anomaly threshold (μ + 2σ) | `Flood_Flag` |

---

## 🔬 Methodology

### Feature Engineering
- **Temporal**: lag features (1, 2, 3, 6, 12 months), rolling statistics (3/6/12-month mean, sum, std), year-over-year comparisons, seasonal cyclical encodings
- **Spatial**: neighbor rainfall statistics (k-NN), upstream rainfall aggregates, spatial rainfall gradients, regional anomalies

### Leakage Control
- All flood-derived aggregates (neighbor flood rate, upstream flood rate, regional flood rate) excluded from features
- Strict temporal split: train 2010–2020, test 2021–2024
- RobustScaler fitted only on training data

### Models
- **Random Forest**: 150 trees, max depth 10, balanced class weights
- **SVM-PSO**: RBF kernel SVM with C and γ tuned via Particle Swarm Optimization (F1 objective)

### Validation
- **Temporal split**: 2010–2020 train → 2021–2024 test
- **Leave-One-Basin-Out (LOBO)**: 34 sequential held-out basins
- **Basin-Grouped 5-Fold CV**: 34 basins split into 5 groups

---

## 📊 Results

### Temporal Evaluation (2021–2024 test set)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| RF (Train) | 0.988 | 0.932 | 0.986 | 0.958 | 0.999 |
| RF (Test) | 0.969 | 0.845 | 0.894 | 0.869 | 0.987 |
| SVM-PSO (Train) | 0.955 | 0.768 | 0.972 | 0.858 | 0.992 |
| SVM-PSO (Test) | 0.862 | 0.454 | 0.931 | 0.610 | 0.910 |

### LOBO Cross-Basin (mean over 34 basins)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| RF | 0.980 | 0.900 | 0.961 | 0.927 | 0.998 |
| SVM-PSO | 0.931 | 0.664 | 0.961 | 0.778 | 0.984 |

### Basin-Grouped 5-Fold CV

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| RF | 0.980 | 0.897 | 0.963 | 0.929 | 0.997 |
| SVM-PSO | 0.930 | 0.666 | 0.964 | 0.787 | 0.983 |

---

## 🚀 Setup & Usage

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/indus-flood-prediction.git
cd indus-flood-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Data collection (requires Google Earth Engine access)
```bash
# Authenticate with GEE first
earthengine authenticate

# Then run the data collection script
python src/data_collection.py
```

### 4. Run the full pipeline
```python
# Or use the notebooks in order:
# notebooks/01_data_collection.ipynb
# notebooks/02_preprocessing.ipynb
# notebooks/03_feature_engineering.ipynb
# notebooks/04_modeling_results.ipynb
```

### 5. Train and evaluate models
```python
from src.preprocessing import preprocess_pipeline
from src.feature_engineering import create_temporal_features, create_spatial_features
from src.models import train_random_forest, train_svm_pso
from src.evaluation import evaluate_classifier, plot_roc_curves

# Load and process data
df = preprocess_pipeline("data/master_with_floodflags.csv")

# Train models
rf_model, rf_proba = train_random_forest(X_train, y_train, X_test)
svm_model, svm_proba = train_svm_pso(X_train, y_train, X_test)
```

---

## 📋 Requirements

See `requirements.txt` for the full list. Key packages:
- `scikit-learn` — ML models and evaluation
- `imbalanced-learn` — SMOTE oversampling
- `geopandas` — shapefile handling
- `geemap` / `earthengine-api` — Google Earth Engine
- `pandas`, `numpy` — data processing
- `matplotlib`, `seaborn` — visualization

---

## 📝 Citation

If you use this work, please cite:

```
Arooj Kausar, "Flood Prediction in the Indus Basin in Pakistan Using Supervised
Machine Learning Models With Cross-Basin Validation," SEECS, NUST, 2025.
```

---

