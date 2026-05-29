# Pyrolysis Activation Energy (Ea) Prediction and Blending Optimization Workflow

This repository provides a self-contained, reproducible data science workflow for predicting pyrolysis **Activation Energy (Ea)**, performing SHAP-based feature attribution, and simulating feedstock blending strategies to optimize and lower activation energy.

The workflow integrates MATLAB-trained deep neural networks (DNN) with Python-based interpretability (SHAP) and stochastic sampling (Monte Carlo) pipelines.

---

## 📁 Project Directory Structure

The repository follows a clean, flattened scientific data layout separating immutable data, model files, modular code scripts, and ignored experimental outputs:

```text
MonteCarlo_bpDNN2Ea/
├── README.md                              # Project documentation (this file)
├── requirements.txt                       # Python environment dependencies
├── .gitignore                             # Ignores local results and binary caches
│
├── data/                                  # 📦 Inputs: Immutable raw datasets
│   └── raw/
│       ├── Municipal_Sludge_Data_cleaned_mean.xlsx   # Feedstock attribute ranges
│       └── US_SewageSludge.xlsx                      # Baseline sewage sludge compositions
│
├── bpDNN4Ea_modelfiles/                   # 🧠 Models: MATLAB trained DNN structure and weights
│   ├── Results_trained.mat                # Serialized model parameters and scaling stats
│   ├── bpDNN4Ea.m                         # Original MATLAB training script reference
│   └── ...                                # Auxiliary MATLAB neural network toolboxes
│
├── scripts/                               # 🛠️ Code: Modular execution Python scripts
│   ├── __init__.py
│   ├── missing_value_handler.py           # Missing data pre-processing utilities
│   ├── shap_analysis_ea.py                # (01) SHAP attribution analysis
│   ├── mc_us_sludge_ea_prediction.py      # (02) Monte Carlo random prediction baseline
│   ├── generate_ea_reduction_blending_strategies.py # (03) Blending strategies and forward simulation
│   └── plot_ea_dependence.py              # (04) Optional: Attribute dependence scatter plotter
│
└── results/                               # 📊 Outputs: Generated experimental reports & figures
    ├── shap_outputs/                      # Attributions, force plots, and beeswarm figures (PNG/SVG/EPS)
    ├── mc_outputs/                        # Stochastic prediction tables (CSV) and uncertainty plots
    └── blending_outputs/                  # Blending strategies, optimization tables, and conversion curves
```

---

## ⚙️ Environment Configuration

Ensure you have a Python 3.9+ environment configured. It is highly recommended to use Conda for virtual environment management:

```bash
# Create and activate a dedicated conda environment
conda create -n bpdnn_shap python=3.11 -y
conda activate bpdnn_shap

# Install required scientific packages
pip install numpy pandas matplotlib seaborn scipy openpyxl shap torch scikit-learn
```

---

## 🚀 Step-by-Step Execution Sequence

The workflow is highly sequential. Each script relies on outputs produced by the preceding step. All path resolutions are dynamically determined from the project root (`pathlib`), enabling **Zero-Configuration** execution directly out of the box.

Change directory to the scripts folder:
```bash
cd scripts
```

### Step 1: SHAP Attribution & Model Interpretation
Analyze the trained MATLAB neural network to quantify feedstock feature contributions to the predicted activation energy (Ea).
```bash
python shap_analysis_ea.py
```
* **Inputs read**: `../bpDNN4Ea_modelfiles/Results_trained.mat`
* **Outputs created**: `../results/shap_outputs/SHAP_Analysis_Ea_Results_[TIMESTAMP]/`
  * Generates vector publication plots (`.svg`, `.eps`) and raster plots (`.png`) for Beeswarm, Feature Importance, Single Instance Force Plot, and Waterfall diagrams.
  * Exports SHAP values to Excel (`shap_values_data.xlsx`) and serializes binary SHAP matrix (`shap_values.npy`).

### Step 2: Monte Carlo Stochastic Prediction
Sample the feedstock attribute space under proximate/ultimate mass-balance constraints to predict absolute activation energy distributions for U.S. sewage sludge.
```bash
python mc_us_sludge_ea_prediction.py --samples 10000 --seed 2025
```
* **Inputs read**:
  * `../bpDNN4Ea_modelfiles/Results_trained.mat` (model & scaling parameters)
  * `../data/raw/Municipal_Sludge_Data_cleaned_mean.xlsx` (jittering ranges)
  * `../data/raw/US_SewageSludge.xlsx` (sewage sludge composition baseline)
* **Outputs created**: `../results/mc_outputs/`
  * `mc_ea_predictions.csv` – raw stochastically predicted Ea values.
  * `mc_ea_distribution.png` / `.svg` – violin + histogram distribution plots.

### Step 3: Blending Optimization & Forward Simulation
Rank feedstocks by their SHAP-deduced Ea-reducing capabilities, and run forward simulations using the MATLAB network wrapper to optimize multi-feedstock mixing ratios.
```bash
python generate_ea_reduction_blending_strategies.py --simulate
```
* **Features**:
  * Automatically scans `../results/shap_outputs/` and matches the **latest** timestamped SHAP analysis run.
  * Automatically reads the baseline `mc_ea_predictions.csv` to calculate relative activation energy reductions ($\Delta\text{Ea}$).
* **Outputs created**: `../results/blending_outputs/`
  * `top_20_feedstocks_ea.csv` – optimal single blending ratios and absolute reductions.
  * `conversion_dependence.png` / `.svg` – curves mapping $\Delta\text{Ea}$ against pyrolysis conversion degrees.
  * `combo_results.csv` / `combo_best.png` – multi-feedstock synergistic mixing optimizations.

### Step 4 (Optional): Plot Ea Attribute Dependence
Generate custom bivariate scatter plots matching SHAP dependence style to visualize how specific features influence activation energy predictions.
```bash
python plot_ea_dependence.py --x Ash_SiO2 --color Ash_Na2O
```
* **Inputs read**: `../results/mc_outputs/mc_ea_predictions.csv`
* **Outputs created**: `../results/mc_outputs/ea_dependence/`

---

## 🎯 Reproducibility & Design Notes

1. **Deterministic Execution**: Global seeds are hardcoded across all Python, NumPy, PyTorch, and SHAP packages to guarantee identical beeswarm shapes, Monte Carlo samples, and optimization curves upon multiple runs.
2. **TrueType Fonts (Type 42)**: Vector graphics exported in `.eps` format use native TrueType font embedding, preventing typeface and outline distortion when importing diagrams directly into Adobe Illustrator for manuscript editing.
3. **Strict Pathing Resolution**: All files use dynamic relative pathing based on Python's `Path(__file__).resolve().parent`. There are no hardcoded machine paths, allowing the workflow to be transferred seamlessly across different operating systems.
