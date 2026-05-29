# Pyrolysis Product Yield Prediction and Synergistic Blending Optimization Workflow

This repository provides a self-contained, high-performance, and reproducible data science workflow for predicting pyrolysis **Product Yields (Biochar, Bioliquid, Biogas)**, performing SHAP-based feature interpretability attributions, and simulating feedstock co-pyrolysis blending recipes using continuous mathematical optimization frameworks on High-Performance Computing (HPC) clusters.

The workflow integrates MATLAB-trained deep neural networks (DNN) with Python-based interpretability (SHAP), multi-core stochastic sampling (Monte Carlo predictions), and Scipy continuous joint-optimization pipelines.

---

## 📁 Project Directory Structure

The repository follows a clean, flattened scientific data layout separating immutable data, model files, modular code scripts, and ignored experimental outputs:

```text
MonteCarlo_bpDNN2Yield/
├── README.md                              # Project documentation (this file)
├── requirements.txt                       # Python environment dependencies
├── .gitignore                             # Ignores local results and binary caches
│
├── data/                                  # 📦 Inputs: Immutable raw datasets
│   └── raw/
│       ├── Municipal_Sludge_Data_cleaned_mean.xlsx   # Feedstock attribute ranges
│       └── US_SewageSludge.xlsx                      # Baseline sewage sludge compositions
│
├── bpDNN4PyroProd_modelfiles/             # 🧠 Models: MATLAB trained DNN structure and weights
│   ├── Results_trained.mat                # Serialized model parameters and scaling stats
│   ├── Script_NaN_detection.m             # Original MATLAB validation script
│   ├── bpDNN4PyroProd.m                   # Original MATLAB training script reference
│   └── ...                                # Auxiliary MATLAB neural network toolboxes
│
├── scripts/                               # 🛠️ Code: Modular execution Python scripts
│   ├── missing_value_handler.py           # Missing data pre-processing utilities
│   ├── shap_analysis_yield.py             # (01) SHAP attribution analysis for Char/Liquid/Gas
│   ├── mc_us_sludge_prediction.py         # (02) 192-Core Monte Carlo random prediction baseline
│   ├── generate_blending_strategies.py    # (03) Blending strategies and continuous optimization
│   ├── test_dependence_plots.py           # (04) Optional: Test script for modified dependency charts
│   ├── visualize_nn_structure.py          # (05) Optional: Neural network structure visualizer
│   ├── run_shap_analysis_slurm.sh         # [HPC] Slurm script for parallel SHAP attributions (192 Cores)
│   ├── run_mc_prediction_slurm.sh         # [HPC] Slurm script for 2M parallel MC predictions (192 Cores)
│   └── run_blending_strategies_slurm.sh   # [HPC] Slurm script for parallel dual-scenario optimization sweeps (192 Cores)
│
└── results/                               # 📊 Outputs: Generated experimental reports & figures (dynamically created)
    ├── shap_outputs/                      # Attributions, force plots, and beeswarm figures (PNG/SVG/EPS)
    ├── mc_outputs/                        # Stochastic prediction tables (CSV) and uncertainty plots
    ├── blending_outputs/                  # Blending recipes, optimization tables, and conversion curves
    └── network_visualization/             # Visualization charts for weights, heatmaps, and summaries
```

---

## ⚙️ Environment Configuration

Ensure you have a Python 3.9+ environment configured. It is highly recommended to use Conda for virtual environment management:

```bash
# Create and activate a dedicated conda environment
conda create -n pyrolysis_model_dnn python=3.11 -y
conda activate pyrolysis_model_dnn

# Install required scientific packages
pip install numpy pandas matplotlib seaborn scipy openpyxl shap
```

---

## 🚀 Step-by-Step Execution Sequence

The workflow is highly sequential. Each script relies on outputs produced by the preceding step. All path resolutions are dynamically determined from the project root (`pathlib`), enabling **Zero-Configuration** execution directly out of the box.

Change directory to the scripts folder:
```bash
cd scripts
```

### Step 1: SHAP Attribution & Model Interpretation
Analyze the trained MATLAB neural network to quantify feedstock feature contributions to Char, Liquid, and Gas yields.
```bash
# For local testing (sequential execution):
python shap_analysis_yield.py

# On Slurm HPC Cluster (automatic 32-core allocation):
sbatch run_shap_analysis_slurm.sh
```
* **Inputs read**: `../bpDNN4PyroProd_modelfiles/Results_trained.mat`
* **Outputs created**: `../results/shap_outputs/SHAP_Analysis_Results_[TIMESTAMP]/`
  * Generates vector publication plots (`.svg`, `.eps`) and raster plots (`.png`) for Beeswarm, Feature Importance, Single Instance Force Plot, and Waterfall diagrams.
  * Exports SHAP values to Excel (`shap_values_data.xlsx`) and serializes binary SHAP matrix (`shap_values.npy`).

### Step 2: 192-Core Monte Carlo Stochastic Prediction
Sample the feedstock attribute space under proximate/ultimate mass-balance constraints to predict absolute yield distributions for U.S. sewage sludge.
```bash
# For local testing (sequential execution):
python mc_us_sludge_prediction.py --samples 10000 --cores 1

# On Slurm HPC Cluster (automatic 192-core allocation):
sbatch run_mc_prediction_slurm.sh
```
* **Inputs read**:
  * `../bpDNN4PyroProd_modelfiles/Results_trained.mat` (model & scaling parameters)
  * `../data/raw/Municipal_Sludge_Data_cleaned_mean.xlsx` (jittering ranges)
  * `../data/raw/US_SewageSludge.xlsx` (sewage sludge composition baseline)
* **Outputs created**: `../results/mc_outputs/`
  * `mc_us_sludge_predictions.csv` – raw stochastically predicted yield values.
  * `mc_us_sludge_uncertainty.png` / `.svg` – violin distribution plots.

### Step 3: Dual-Scenario Blending Optimization (Scipy Continuous Optimization)
Rank feedstocks by their SHAP-deduced yield-increasing capabilities, and run forward simulations using the MATLAB network wrapper to optimize multi-feedstock mixing ratios using continuous math optimization.
```bash
# For local testing (sequential execution):
python generate_blending_strategies.py --simulate --method scipy --cores 1

# On Slurm HPC Cluster (automatic 192-core dual-scenario sweeps):
sbatch run_blending_strategies_slurm.sh
```
* **Features**:
  * Scans `../results/shap_outputs/` and matches the **latest** timestamped SHAP analysis run.
  * Reads the baseline `mc_us_sludge_predictions.csv` to calculate relative yield gain.
  * Enforces exact **sludge capacity constraints** and **single-additive caps** under SLSQP solvers.
* **Outputs created**: `../results/blending_outputs_[50|20]/`
  * `top_20_feedstocks_yield.csv` – optimal single blending ratios and absolute reductions.
  * `conversion_dependence.png` / `.svg` – curves mapping $\Delta\text{Yield}$ against temperature.
  * `combo_results.csv` / `combo_best.png` – multi-feedstock synergistic mixing optimizations.

---

## 🎯 Reproducibility & Design Notes

1. **Deterministic Execution**: Global seeds and child SeedSequences are used across all Python, NumPy, PyTorch, and SHAP packages to guarantee identical beeswarm shapes, Monte Carlo samples, and optimization curves upon multiple runs.
2. **Sparsity Protection filter**: Continuous optimization sweeps naturally collapse boundary variables. A mathematical safeguard ($\ge 0.1\%$ mixing ratio) blocks degenerate representations, forcing true $k$-component co-pyrolysis blends.
3. **Strict Pathing Resolution**: All files use dynamic relative pathing based on Python's `Path(__file__).resolve().parent`. There are no hardcoded machine paths, allowing the workflow to be transferred seamlessly across different operating systems.
