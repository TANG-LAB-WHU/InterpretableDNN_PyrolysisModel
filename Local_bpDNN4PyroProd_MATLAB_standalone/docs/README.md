# SHAP Analysis for Pyrolysis Model

This folder contains scripts for training and analyzing a neural network model for pyrolysis product prediction using SHAP (SHapley Additive exPlanations) values.

## Overview

SHAP values provide a unified approach to explaining the output of machine learning models by assigning each feature a value for its contribution to a particular prediction. This implementation applies SHAP analysis to a backpropagation deep neural network (bpDNN) model for predicting pyrolysis product yields.

## Scripts

The analysis consists of the following scripts:

1. `run_analysis.m` - Main script that orchestrates the entire workflow
2. `debug_run_analysis.m` - Debug version that runs with reduced epochs (60 instead of 6000) for faster testing
3. `calc_shap_values.m` - Calculates SHAP values for the trained model using KernelSHAP
4. `plot_shap_results.m` - Generates visualizations and analysis of SHAP values

## How to Run

Simply execute the main script in MATLAB:

```matlab
% For full analysis (6000 epochs):
run('GPM_SHAP_matlab/Scripts/run_analysis.m')

% For debugging/testing (60 epochs):
run('GPM_SHAP_matlab/Scripts/debug_run_analysis.m')
```

The script will:
1. Train the neural network model using files in the `bpDNN4PyroProd_modelfiles` folder
2. Calculate SHAP values for all samples and targets
3. Generate visualizations including beeswarm plots, feature importance plots, and dependence plots
4. Save results in the `GPM_SHAP_matlab/Results/SHAP_Analysis` folder

## Output

The analysis generates the following outputs:

- **Feature Importance Plots**: Bar charts showing the mean absolute SHAP values for each feature
- **Beeswarm Plots**: Shows the distribution of SHAP values for each feature, colored by feature value
- **Dependence Plots**: Shows how SHAP values depend on feature values
- **Individual Sample Explanations**: Waterfall plots explaining individual predictions
- **Summary Report**: Text file summarizing the key findings

All visualizations are saved as both PNG and FIG files for further analysis.

## Performance Optimization

The script uses MATLAB's parallel computing capabilities to accelerate SHAP value calculations. The main optimizations include:

- Parallel processing using `parfor` loops
- Efficient matrix operations
- Selective sampling for background distribution
- Pre-allocation of arrays for better memory usage

## Debug Mode

The debug version (`debug_run_analysis.m`) modifies the training process to use only 60 epochs instead of the default 6000. This allows for much faster execution during debugging and code testing. The script works by:

1. Creating a temporary modified copy of the model training script
2. Changing the epoch number from 6000 to 60
3. Running the modified script
4. Cleaning up the temporary file after execution

The resulting model will be less accurate but sufficient for testing SHAP analysis functionality.

## References

1. Lundberg, S.M., Lee, S.I. (2017). A Unified Approach to Interpreting Model Predictions. Advances in Neural Information Processing Systems 30.
2. SHAP documentation: [https://shap.readthedocs.io/](https://shap.readthedocs.io/) 