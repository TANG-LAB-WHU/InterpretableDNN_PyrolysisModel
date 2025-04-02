# GPM_SHAP: Neural Network and SHAP Analysis for Pyrolysis Model

This repository contains a MATLAB implementation for training and analyzing neural network models for pyrolysis product prediction using SHAP (SHapley Additive exPlanations) values.

## Project Structure

```
├── GPM_SHAP_matlab/
│   ├── Scripts/             # Analysis scripts
│   │   ├── run_analysis.m   # Main analysis script
│   │   ├── debug_run_analysis.m  # Debug version with reduced epochs
│   │   ├── calc_shap_values.m    # SHAP value calculation
│   │   ├── plot_shap_results.m   # Visualization generation
│   │   └── ...
│   ├── Results/             # Output results
│   │   ├── Training/        # Full training results
│   │   ├── Training_Debug/  # Debug training results
│   │   ├── SHAP_Analysis/   # Full SHAP analysis results
│   │   └── SHAP_Analysis_Debug/  # Debug SHAP analysis results
│   └── run_analysis_log.txt # Analysis log file
├── bpDNN4PyroProd_modelfiles/  # Neural network model files
│   ├── bpDNN4PyroProd.m     # Main neural network model
│   ├── nn*.m                # Neural network utility functions
│   ├── RawInputData.xlsx    # Input data for model training
│   └── ...
└── README.md                # This file
```

## Overview

This project provides a framework for analyzing how different features affect the prediction of pyrolysis products using a backpropagation deep neural network (bpDNN) model. The SHAP values provide insights into the contribution of each feature to the model's predictions.

## How to Run

There are two main execution modes:

### Full Analysis (6000 epochs)

```matlab
run('GPM_SHAP_matlab/Scripts/run_analysis.m')
```

### Debug/Testing Mode (60 epochs)

```matlab
run('GPM_SHAP_matlab/Scripts/debug_run_analysis.m')
```

## Process Flow

1. **Model Training**: The neural network model is trained using the files in the `bpDNN4PyroProd_modelfiles` folder
2. **SHAP Calculation**: SHAP values are calculated for all samples and targets using KernelSHAP
3. **Visualization**: The script generates:
   - Feature importance plots
   - Beeswarm plots
   - Dependence plots
   - Individual sample explanations
4. **Results Storage**: All outputs are saved in the `GPM_SHAP_matlab/Results/` directory

## Performance Optimization

The implementation uses MATLAB's parallel computing capabilities for:
- Parallel processing using `parfor` loops
- Efficient matrix operations
- Selective sampling for background distribution
- Pre-allocated arrays for memory optimization

## References

1. Lundberg, S.M., Lee, S.I. (2017). A Unified Approach to Interpreting Model Predictions. Advances in Neural Information Processing Systems 30.
2. SHAP documentation: [https://shap.readthedocs.io/](https://shap.readthedocs.io/) 