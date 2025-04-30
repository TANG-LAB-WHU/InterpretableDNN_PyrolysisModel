# GPM_SHAP: Neural Networks and SHAP Analysis for Pyrolysis Models

This project provides a MATLAB implementation for training and analyzing neural network models for pyrolysis product prediction with interpretability analysis using SHAP (SHapley Additive exPlanations) values.

## Project Overview

This project establishes a framework for analyzing how different features impact the prediction of pyrolysis products using a backpropagation deep neural network (bpDNN) model. SHAP values provide insights into the contribution of each feature to the model's predictions, helping to understand the decision-making process of the model.

## Project Structure

```
GPM_SHAP_FinalVersion/
+-- data/                  # Input data directory
|   +-- raw/               # Raw input data files
|   +-- processed/         # Preprocessed data
|
+-- results/               # Results directory
|   +-- training/          # Full training results
|   |   +-- Figures/       # Figures generated during full training
|   |
|   +-- debug/             # Debug training results (with fewer epochs)
|   |   +-- Figures/       # Figures generated during debug training
|   |
|   +-- optimization/      # Hyperparameter optimization results
|   |   +-- config_*/      # Individual configuration results
|   |   +-- best_model.mat # Best model from optimization
|   |
|   +-- best_model/        # Best model training results
|   |   +-- Figures/       # Figures from best model training
|   |
|   +-- analysis/          # SHAP analysis results
|       +-- full/          # Full analysis results
|       |   +-- data/      # SHAP data for full analysis
|       |   +-- figures/   # SHAP visualization figures for full analysis
|       |   +-- shap_analysis_results.xlsx # Excel export of SHAP analysis
|       |
|       +-- debug/         # Debug analysis results
|           +-- data/      # SHAP data for debug analysis
|           +-- figures/   # SHAP visualization figures for debug analysis
|           +-- shap_analysis_results.xlsx # Excel export of SHAP analysis
|
+-- src/                   # Source code directory
|   +-- model/             # Neural network model implementation
|   |   +-- bpDNN4PyroProd.m  # Main neural network model
|   |   +-- nnbp.m            # Backpropagation implementation with adaptive learning rate
|   |   +-- nncheckgrad.m     # Gradient checking utility
|   |   +-- nnconfigure.m     # Network configuration
|   |   +-- nncreate.m        # Network creation
|   |   +-- nneval.m          # Network evaluation
|   |   +-- nnff.m            # Feed-forward computation
|   |   +-- nnfindbest.m      # Find best network model
|   |   +-- nninit.m          # Network initialization
|   |   +-- nnpostprocess.m   # Post-processing utility
|   |   +-- nnpredict.m       # Prediction function
|   |   +-- nnprepare.m       # Data preparation
|   |   +-- nnpreprocess.m    # Pre-processing utility
|   |   +-- nnstopcriteria.m  # Training stop criteria
|   |   +-- nntrain.m         # Neural network training
|   |   +-- nnupdatefigure.m  # Training visualization
|   |
|   +-- scripts/           # Analysis and utility scripts
|   |   +-- run_analysis.m            # Main analysis script
|   |   +-- debug_run_analysis.m      # Ultra-fast debug version (20 epochs)
|   |   +-- optimize_hyperparameters.m # Hyperparameter optimization
|   |   +-- train_best_model.m        # Train model with best hyperparameters
|   |   +-- calc_shap_values.m        # SHAP value calculation
|   |   +-- export_shap_to_excel.m    # Export SHAP values to Excel
|   |   +-- script4PDP.m              # PDP script
|   |
|   +-- visualization/     # Plotting and visualization scripts
|       +-- plot_shap_results.m           # SHAP results plotting
|       +-- plot_shap_beeswarm.m          # Beeswarm plot generation
|       +-- plot_shap_original_summary.m  # Original summary plotting
|       +-- fix_colorbar_style.m          # Colorbar style adjustment
|
+-- docs/                  # Documentation
+-- output/                # Process logs and outputs
    +-- debug_analysis/    # Log files for debug runs
    +-- full_analysis/     # Log files for full runs
```

## Workflow

The project follows this workflow:

1. **Hyperparameter Optimization**:
   ```matlab
   cd src/scripts
   optimize_hyperparameters
   ```
   This searches through combinations of hyperparameters including:
   - Learning rate and momentum
   - Learning rate increase/decrease factors
   - Network architectures (hidden layer structure)
   - Transfer functions (for hidden and output layers)
   - Training data division strategy (TT or TVT)
   - Data partition ratios
   
   Results are saved to `results/optimization/` with the best configuration in `best_model.mat`.

2. **Best Model Training**:
   ```matlab
   cd src/scripts
   train_best_model
   ```
   Using the optimal hyperparameters, this trains a full model with all epochs and saves it to:
   - `results/best_model/best_model.mat` (primary location)
   - `results/training/trained_model.mat` (copy for analysis scripts)

3. **Full Analysis**:
   ```matlab
   cd src/scripts
   run_analysis
   ```
   This script:
   - Automatically checks for and uses the best trained model if available
   - Performs SHAP analysis on the model to explain predictions
   - Saves results to `results/analysis/full/`
   - Exports SHAP values to Excel for further analysis

4. **Debug Mode** (Ultra-fast testing):
   ```matlab
   cd src/scripts
   debug_run_analysis
   ```
   Similar to full analysis but with significantly fewer epochs (20 instead of 6000) for ultra-fast execution:
   - Runs comprehensive diagnostics and logs detailed information
   - Creates a detailed execution log in `debug_run_analysis_log.txt`
   - Performs integrity checks on all output files
   - Reports execution time and provides clear next steps
   - Saves results to `results/analysis/debug/`
   - Also exports SHAP values to Excel

All scripts maintain consistent file paths, automatically create necessary directories, and store all outputs in the results directory.

## HPC Cluster Submission (SLURM)

For running analyses on HPC clusters using SLURM job scheduler, two batch scripts are provided:

### 1. Debug Analysis Script (run_debug_analysis.sh)

This script submits a job to run the debug version of the analysis:

```bash
sbatch run_debug_analysis.sh
```

Features:
- Allocates 8 cores for faster debugging
- Sets a 2-hour time limit
- Automatically creates all required directories
- Exports SHAP values to Excel after analysis
- Checks for required data files before running
- Validates model files exist
- Creates detailed job summary file

### 2. Full Analysis Script (run_full_analysis.sh)

This script submits a job to run the full version of the analysis:

```bash
sbatch run_full_analysis.sh
```

Features:
- Allocates 16 cores for full parallel processing
- Sets a 24-hour time limit
- Automatically creates all required directories
- Exports SHAP values to Excel after analysis
- Checks for previous optimization results
- Creates detailed job summary file

Both scripts include detailed logging and error handling, making them suitable for unattended execution on HPC clusters.

## Directory Organization

Results are organized into distinct directories:

1. **Optimization Results**: 
   - Located in `results/optimization/`
   - `config_XXXX/` folders contain individual configuration results
   - `best_model.mat` contains the best configuration
   - `best_configuration.txt` provides a human-readable summary

2. **Best Model Results**:
   - Located in `results/best_model/`
   - Contains the model trained with optimal hyperparameters
   - Includes training visualizations in `Figures/`

3. **Training Results**: 
   - Located in `results/training/` (full mode) or `results/debug/` (debug mode)
   - Includes model checkpoints and training visualizations

4. **SHAP Analysis Results**:
   - Full mode: `results/analysis/full/`
   - Debug mode: `results/analysis/debug/`
   - Each includes:
     - `data/` for SHAP values data files
     - `figures/` for visualization figures
     - `shap_analysis_results.xlsx` Excel export with multiple sheets containing:
       - Feature importance rankings
       - Raw SHAP values for all samples
       - Feature importance metrics
       - Summary information

5. **Output Logs**:
   - Debug mode: `output/debug_analysis/`
   - Full mode: `output/full_analysis/`
   - Contains detailed execution logs and error information

## Hyperparameter Optimization

The hyperparameter optimization process tests multiple combinations of:
- Learning rate (lr)
- Momentum coefficient (mc)
- Learning rate increase factor (lr_inc)
- Learning rate decrease factor (lr_dec)
- Hidden layer structure
- Transfer functions for hidden and output layers
- Training strategy (TT: Training-Testing or TVT: Training-Validation-Testing)
- Data division ratios

The best configuration is automatically saved and can be used for full model training.

### Training Strategies

- **TVT (Training-Validation-Testing)**: Data is divided into three sets - training data used for model learning, validation data used for early stopping and hyperparameter selection, and testing data for final performance evaluation.

- **TT (Training-Testing)**: Data is divided into just two sets - training data used for model learning and testing data for performance evaluation. This strategy is useful when you want to maximize the amount of data used for training while still having a test set for evaluation.

### Transfer Functions

The system supports a wide range of transfer functions for both hidden and output layers:

**Hidden Layer Options:**
- `logsig` - Logarithmic sigmoid: f(x) = 1/(1+e^(-x)), range [0,1]
- `tansig` - Hyperbolic tangent sigmoid: f(x) = 2/(1+e^(-2x))-1, range [-1,1]
- `poslin` - Positive linear (ReLU): f(x) = max(0,x)
- `radbas` - Radial basis function: f(x) = e^(-x^2)
- `softmax` - Softmax function: f(x_i) = e^(x_i)/sum(e^(x_j))
- `elliotsig` - Elliott sigmoid: f(x) = x/(1+|x|), computationally efficient

**Output Layer Options:**
- `purelin` - Pure linear: f(x) = x, unbounded output
- `tansig` - Hyperbolic tangent sigmoid: bounded output [-1,1]
- `logsig` - Logarithmic sigmoid: bounded output [0,1]
- `poslin` - Positive linear (ReLU): f(x) = max(0,x)
- `satlin` - Saturating linear: f(x) = 0 if x<0, x if 0<=x<=1, 1 if x>1

Different combinations work best for different problems. For regression problems like our pyrolysis prediction, `tansig` or `logsig` for hidden layers with `purelin` for the output layer is often effective.

## Core Files Description

1. **Neural Network Model Files**:
   - `bpDNN4PyroProd.m`: Main neural network model implementation
   - `nntrain.m`: Neural network training function
   - `nnpredict.m`: Prediction function
   - `nnff.m`: Feed-forward function
   - `nnbp.m`: Backpropagation implementation with adaptive learning rate

2. **Optimization and Training Scripts**:
   - `optimize_hyperparameters.m`: Systematically tests different hyperparameter combinations
   - `train_best_model.m`: Trains a full model with the best hyperparameters found

3. **Analysis Scripts**:
   - `run_analysis.m`: Main script for complete analysis pipeline
   - `debug_run_analysis.m`: Ultra-fast debugging with diagnostic output and integrity checks
   - `calc_shap_values.m`: SHAP value calculation for interpreting model decisions
   - `export_shap_to_excel.m`: Exports SHAP analysis to Excel with multiple informative sheets
   - `plot_shap_results.m`: Generates various visualization outputs from SHAP analysis
   - `plot_shap_beeswarm.m`: Creates beeswarm plots showing feature importance distribution
   - `fix_colorbar_style.m`: Unifies colorbar style across all images

## Performance Optimization

This implementation leverages MATLAB's parallel computing capabilities for:
- Parallel processing with `parfor` loops
- Efficient matrix operations
- Selective sampling of background distributions
- Array preallocation to optimize memory usage
- Adaptive learning rate using lr_inc and lr_dec parameters

## Important Notes

1. All necessary directory structures are created automatically by the scripts
2. SHAP analysis results are stored in the `results/analysis/` directory, not in the source directory
3. Excel exports of SHAP analysis are automatically created in the corresponding results directory
4. Ensure MATLAB is installed and the correct working directory is set before running scripts
5. Hyperparameter optimization may require significant time; consider running overnight
6. Full training with optimal parameters provides the best results but requires more time
7. Debug mode is recommended for quickly testing changes to the analysis pipeline (20 epochs)
8. The `run_analysis.m` and `debug_run_analysis.m` scripts automatically check for and use the best model if available
9. All files contain only ASCII characters for better portability and compatibility
10. Both TT and TVT training strategies are fully supported with appropriate visualizations
11. The workflow maintains consistency between all steps and preserves optimization results
12. Detailed diagnostic logging is available in debug mode to quickly identify any issues

## Running on NCSA ICC

This code is configured to run on the NCSA Illinois Campus Cluster (ICC). The batch scripts are set up to:
- Load the MATLAB module (version 24.1)
- Configure appropriate parallel computing settings for the cluster
- Use SLURM job scheduling parameters
- Create a dedicated MATLAB temporary directory in scratch space
- Clean up temporary files after completion

For more details on running MATLAB jobs on the ICC, see the [NCSA ICC MATLAB documentation](https://docs.ncsa.illinois.edu/systems/icc/en/latest/user_guide/software.html#matlab).

## Error Handling and Workflow Behavior

Important note about the workflow behavior:

- The workflow will stop and report errors if any step fails.
- SHAP analysis will not be executed if model training fails to avoid misleading results.
- If hyperparameter optimization fails, the workflow will attempt to use the best available results but will stop if no valid results are found.
- If model training fails (for example, due to a matrix dimension mismatch error), the workflow will stop immediately and will not proceed to SHAP analysis.

## Error Log Locations

If errors occur, check the following locations for error messages:

- Main workflow logs: `output/full_analysis/full_[JOB_ID].out` and `output/full_analysis/full_[JOB_ID].err`
- Training failure log: `results/training_failure_log.txt`
- MATLAB execution log: `run_optimized_model_log.txt`
- Summary file: `output/full_analysis/full_summary_[JOB_ID].txt`

## Training Strategies

The code supports two different training strategies:

1. **TVT (Train-Validation-Test)**: Uses separate validation and test sets.
2. **TT (Train-Test)**: Uses only training and test sets (no validation set).

Both strategies are handled correctly by the optimized model script, which includes safety checks to prevent dimension mismatch errors. If using the TT strategy, the validation ratio is explicitly set to 0 to prevent matrix dimension issues.

## Output Files

After successful execution, check these key output files:

- Optimization results: `results/optimization/best_model.mat`
- Best model: `results/best_model/best_model.mat`
- SHAP results: `results/analysis/full/data/shap_results.mat`
- SHAP Excel output: `results/analysis/full/shap_analysis_results.xlsx`
- SHAP figures: `results/analysis/full/figures/`
- Summary report: `output/full_analysis/full_summary_[JOB_ID].txt`