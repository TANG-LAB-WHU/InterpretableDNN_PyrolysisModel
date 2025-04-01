# Pyrolysis Model Training and SHAP Analysis

Deep Neural Network (DNN) for predicting pyrolysis product yields, with SHAP (SHapley Additive exPlanations) analysis for feature importance.

## Project Overview

This project aims to build a deep neural network model to predict product yields from pyrolysis biomass, and uses SHAP analysis to explain model predictions and feature importance. This analysis helps understand how different factors (such as temperature, heating rate, biomass composition, etc.) affect the product distribution in the pyrolysis process. Additionally, significance analysis is used to identify key feature variables that have the greatest impact on model output.

The project consists of three key steps:
1. **Model Parameter Optimization**: Using grid search and cross-validation to find optimal neural network parameters
2. **Model Training**: Training the deep neural network model with optimized parameters
3. **SHAP Analysis**: Performing explanatory analysis on the trained model to identify key features and their impacts

## Directory Structure

```
.
├── data/                     # Data directory
│   ├── raw/                  # Raw data files
│   │   └── RawInputData.xlsx # Pyrolysis experiment data
│   └── processed/            # Processed data
│       └── CopyrolysisFeedstock.mat # Preprocessed pyrolysis data
│
├── src/                      # Source code
│   ├── matlab/               # MATLAB source code
│   │   ├── bpDNN4PyroProd.m  # Main DNN model training script
│   │   ├── optimize_model_params.m # Model parameter optimization script
│   │   ├── train_with_optimal_params.m # Script for training with optimal parameters
│   │   ├── nn*.m             # Neural network helper functions
│   │   └── extract_vars_for_shap.m # Extract variables for SHAP analysis
│   └── python/               # Python source code
│       └── shap_analysis.py  # SHAP analysis script
│
├── scripts/                  # Run scripts
│   └── run_complete_analysis.sh # Script to run complete analysis on ICC cluster
│
├── output/                   # Output directory
│   ├── figures_tracking_training_process/ # Visualizations of neural network training process
│   ├── optimization_results/ # Model parameter optimization results (created during optimization)
│   │   ├── best_model.mat    # Best model parameters
│   │   ├── parameter_search_results.csv # Parameter search results
│   │   ├── top5_models.csv   # Top 5 model parameters
│   │   └── optimization_summary.png # Optimization results visualization
│   ├── shap_analysis/        # SHAP analysis results
│   │   ├── trained_model.mat # Trained model
│   │   ├── shap_values.csv   # SHAP values (CSV format)
│   │   ├── shap_values.npy   # SHAP values (NumPy format)
│   │   ├── *.png             # SHAP visualization images
│   │   └── significant_features/ # Dedicated analysis for significant features
│   └── results/              # Trained model parameters and backups
│       ├── Results_trained.mat # Complete results after training
│       ├── bpDNN4PyroProd.m  # Copy of training script used to generate results
│       ├── Result_needed.zip # Zipped backup of essential files
│       └── Figures/          # Backup copies of training images
│
└── docs/                     # Documentation
```

## Installation and Dependencies

### MATLAB Dependencies
- MATLAB R2024a or higher
- Neural Network Toolbox
- Parallel Computing Toolbox (recommended for acceleration)
- Statistics and Machine Learning Toolbox (for parameter optimization)

### Python Dependencies
- Python 3.11+
- NumPy
- Matplotlib
- SciPy
- pandas
- scikit-learn
- SHAP
- joblib

You can install Python dependencies with the following commands:
```bash
conda create -n shap_env python=3.11
conda activate shap_env
conda install -c conda-forge shap scikit-learn pandas matplotlib scipy joblib
```

## Workflow

The complete workflow of the project includes the following steps:

1. **Model Parameter Optimization**:
   - Use grid search to explore different combinations of neural network parameters
   - Perform cross-validation to evaluate performance of each parameter set
   - Select parameter configuration with best performance on validation set

2. **Model Training**:
   - Train the complete model using optimized parameter configuration
   - Evaluate model performance on training, validation, and test sets
   - Save the trained model and visualization results

3. **SHAP Analysis**:
   - Extract parameters from the trained model
   - Build a random forest surrogate model for SHAP analysis
   - Generate feature importance and dependency charts
   - Perform significance analysis to identify key features

## Usage

### Running Complete Analysis on ICC Cluster

1. Submit job script:
```bash
sbatch scripts/run_complete_analysis.sh
```

2. Skip parameter optimization (if previously completed):
```bash
SKIP_OPTIMIZATION=true sbatch scripts/run_complete_analysis.sh
```

3. After job completion, results will be saved in the respective output directories.

### Running Individual Steps

1. Parameter optimization:
```matlab
cd src/matlab
optimize_model_params
```

2. Training model with optimized parameters:
```matlab
cd src/matlab
train_with_optimal_params
```

3. Training model directly without optimization:
```matlab
cd src/matlab
bpDNN4PyroProd
```

4. Extracting variables for SHAP analysis:
```matlab
cd src/matlab
extract_vars_for_shap
```

5. Running SHAP analysis:
```bash
python src/python/shap_analysis.py
```

## Model Parameter Optimization

This project uses grid search combined with cross-validation to optimize neural network parameters. The optimized parameters include:

- **Hidden layer configuration**: Exploring different numbers and sizes of hidden layers
- **Learning rate**: Adjusting model learning speed
- **Momentum coefficient**: Optimizing training convergence performance
- **Activation function**: Testing different types of neuron activation functions

The parameter optimization process uses statistical analysis to find the optimal parameter combination, generates detailed performance reports, and automatically saves the best model configuration. This optimization process greatly improves the accuracy and stability of the model.

## Output Files

### Parameter Optimization Results
- `output/optimization_results/best_model.mat`: Saved best parameter configuration and model
- `output/optimization_results/parameter_search_results.csv`: Performance results for all parameter combinations
- `output/optimization_results/top5_models.csv`: Top 5 parameter configurations by performance
- `output/optimization_results/optimization_summary.png`: Parameter optimization results visualization

### Basic SHAP Analysis Images
- `output/shap_analysis/shap_violin_plot.png`: Violin plot of SHAP value distribution
- `output/shap_analysis/shap_beeswarm_plot.png`: Beeswarm plot of SHAP values
- `output/shap_analysis/shap_dependence_plot_feature_*.png`: Dependency plots for important features
- `output/shap_analysis/shap_waterfall_plot.png`: Waterfall plot for sample prediction
- `output/shap_analysis/feature_importance.csv`: Feature importance ranking table

### Significance Analysis Images
- `output/shap_analysis/significant_features_importance.png`: Bar chart of significant feature importance
- `output/shap_analysis/significant_features_violin_plot.png`: Violin plot of SHAP values for significant features
- `output/shap_analysis/significant_features_beeswarm_plot.png`: Beeswarm plot of SHAP values for significant features
- `output/shap_analysis/significant_features_pie_chart.png`: Pie chart of relative importance for significant features
- `output/shap_analysis/significant_features_interaction_heatmap.png`: Interaction heatmap between significant features
- `output/shap_analysis/significant_features_importance.csv`: Significant feature importance data table
- `output/shap_analysis/significant_features/dependence_plot_*.png`: Individual dependency plots for each significant feature

### Data Files
- `output/shap_analysis/shap_values.npy`: Saved SHAP values (NumPy format) for subsequent analysis
- `output/shap_analysis/shap_values.csv`: Saved SHAP values (CSV format) for easy viewing and processing
- `output/figures_tracking_training_process/Fig*.fig`: MATLAB figure files of neural network training process
- `output/figures_tracking_training_process/Fig*.tif`: High-quality TIF format images of neural network training process
- `output/results/Results_trained.mat`: Complete training results backup
- `output/results/bpDNN4PyroProd.m`: Copy of the training script used to generate results
- `output/results/Result_needed.zip`: Zipped backup of essential files (training results, figures, and script)
- `output/results/Figures/Fig*.fig`: Backup copies of training images

## Significance Analysis Method

This project determines significant features through the following steps:

1. Calculate the mean absolute value of SHAP values for all features as a measure of feature importance
2. Calculate the mean and standard deviation of feature importance
3. Set the significance threshold as `mean importance + standard deviation`
4. Select features with importance above the threshold as significant features
5. Generate specialized visualizations and analysis for significant features

Significance analysis helps identify key features that have the greatest impact on model predictions, thereby simplifying model interpretation and focusing on the most important variables.

## References

- [SHAP Documentation](https://shap.readthedocs.io/en/latest/index.html)
- [SHAP Shapley Values Introduction](https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html)
- [Illinois Campus Cluster Documentation](https://docs.ncsa.illinois.edu/systems/icc/en/latest/) 