# Improved Pyrolysis Kinetic Parameter Optimization

This folder contains improved scripts for the optimization of kinetic parameters in pyrolysis modeling, specifically for sewage sludge samples.

## Overview of Improvements

The optimization process has been enhanced to address several issues:

1. **Fixed Index Bounds Error**: Resolved the "Index exceeds the number of array elements" error that occurred in the original script.
2. **Improved Matrix Handling**: Corrected the orientation and dimension handling of the G_alpha matrix.
3. **Enhanced Error Handling**: Added robust error checking and fallback mechanisms to prevent failures during optimization.
4. **Interactive Mode**: Added an interactive script that allows pausing between activation energy iterations.
5. **Better Logging**: Improved logging to track optimization progress and diagnose issues.
6. **Adaptive Bounds**: Enhanced the adaptive bounds generation for more reliable optimization results.

## How to Run the Optimization

### Option 1: Using the PowerShell Script (Windows)

For Windows users, a PowerShell script has been provided to simplify the execution:

1. Open PowerShell or Command Prompt
2. Navigate to the PKP_validation directory
3. Run: `.\run_optimization_windows.ps1`
4. Follow the on-screen prompts to select the optimization mode

### Option 2: Running Directly in MATLAB

You can also run the scripts directly in MATLAB:

1. Open MATLAB
2. Navigate to the `2_GA4Tsolution_PARA` directory
3. Run one of the following commands:
   - For interactive mode: `run_optimization_interactive`
   - For full optimization: `GA4Tsolution_sludge_para_improved`

## Available Scripts

- **run_optimization_interactive.m**: Runs the optimization with prompts to continue between iterations
- **GA4Tsolution_sludge_para_improved.m**: Improved version of the original script with better error handling
- **multiStageOptimization_improved.m**: Enhanced optimization routine with better error recovery
- **generateAdaptiveBounds_improved.m**: Improved boundary generation for optimization variables
- **Conversion_fixed.m**: Fixed version of the reaction model conversion function

## Checking Results

After running the optimization, results will be available in the following locations:

1. **optimization_diagnostics/**: Contains detailed logs of the optimization process
2. **Results_kinetics.mat**: The final optimization results
3. **Results_kinetics_Ea_X.mat**: Interim results for each activation energy (only with interactive mode)

## Troubleshooting

If you encounter issues during the optimization:

1. Check the log files in the `optimization_diagnostics` directory
2. Ensure that all required input data is available (Ea_prediction_results.mat, alphaTG_exp.mat)
3. Verify that the parallel computing toolbox is installed in MATLAB
4. Make sure you have sufficient memory for parallel processing

## Notes on Parallel Processing

The optimization uses MATLAB's parallel processing capabilities. The number of workers used depends on your system configuration. You can modify the parpool settings in the scripts if needed.
