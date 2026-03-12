# bpDNN4Ea: Neural Network for Activation Energy Prediction

This project provides a MATLAB implementation for training and analyzing neural network models to predict the activation energy (Ea) of biomass pyrolysis processes.

## Project Overview

This project establishes a framework for predicting activation energy using a backpropagation deep neural network (bpDNN). It includes scripts for model training, kinetic parameter validation, and visualization of training performance against experimental data.

## Project Structure

```
bpDNN4Ea_src/
+-- Validation_with_ExpData/   # Validation against experimental datasets
|   +-- Results.mat            # Stored validation results
|   +-- PKP_validation.m       # Parallel Kinetic Parameter validation script
|   +-- simulatedTG_*.fig      # Visualization of simulated TG curves
|
+-- Dataset4Ea_new.xlsx        # Main Excel dataset for Ea prediction
+-- dataset4Ea.mat             # MATLAB formatted dataset
+-- bpDNN4Ea.m                 # Main script for Ea prediction and model training
+-- RunPKP.m                   # Script to run Kinetic Parameter analysis
+-- Conversion.m               # Calculation of biomass conversion rates
+-- Enthalpy.m                 # Utility for enthalpy calculations
+-- MC_integral.m              # Monte Carlo integration for kinetics
+-- normalizedweight.m         # Weight normalization utility
+-- nntrain.m                  # Core neural network training function
+-- nnpredict.m                # Core prediction function
+-- nnbp.m                     # Backpropagation implementation
+-- nn*.m                      # Various neural network helper scripts
+-- *.fig                      # Training performance and regression plots
```

## Workflow

1. **Data Preparation**:
   The system uses `Dataset4Ea_new.xlsx` for training. Data is preprocessed using `nnprepare.m` and `nnpreprocess.m`.

2. **Model Training**:
   Run `bpDNN4Ea.m` to train the neural network for activation energy prediction. The script leverages the underlying `nn*` toolbox for:
   - Network configuration and creation
   - Feed-forward and backpropagation
   - Training visualization and stop criteria

3. **Kinetic Parameter Validation**:
   Use `RunPKP.m` or the scripts within `Validation_with_ExpData/` to validate the predicted kinetic parameters against experimental TG (Thermogravimetric) data.

4. **Visualization**:
   The project generates several plots to evaluate performance:
   - `bpDNN4Ea_WholeTrainingPerformance.fig`: Overall training metrics.
   - `bpDNN4Ea_WholeTrainingPerformance_Regression.fig`: Regression analysis between predicted and actual Ea.

## Core Files Description

1. **Main Prediction Scripts**:
   - `bpDNN4Ea.m`: Main entry point for training the activation energy model.
   - `RunPKP.m`: Entry point for kinetic parameter validation workflows.

2. **Calculation Utilities**:
   - `Conversion.m`: Handles the conversion of raw data into reaction progress (alpha).
   - `MC_integral.m`: Performs Monte Carlo integration for kinetic analysis.
   - `Enthalpy.m`: Calculates thermal properties related to the pyrolysis process.

3. **Neural Network Engine**:
   - `nntrain.m`, `nnbp.m`, `nnff.m`: The core engine for backpropagation and training.
   - `nnconfigure.m`, `nncreate.m`, `nninit.m`: Network setup and initialization.
   - `nnpredict.m`, `nneval.m`: Model application and performance evaluation.

## Important Notes

1. Ensure all `nn*.m` helper scripts are in the same directory as the main scripts.
2. The `Validation_with_ExpData` folder contains its own set of experimental comparisons and simulated results.
3. All file paths are relative to the project root for portability.
4. The implementation supports adaptive learning rates and multiple transfer functions (logsig, tansig, purelin, etc.).