# Results Analysis & Post-Processing

This document provides a guide for post-processing, analyzing, and plotting the output datasets from the Modular Adaptive Multi-Mechanism Pyrolysis Simulation.

---

## I. Output Directory Structure

Upon completion, all datasets and publication-quality figures are exported to the results directory (e.g., `CornStover_SimulatedKinetics/Results-MultiMechanism/`):

### 1. Data Files
*   **`TG_curve_w_inf_endpoint.csv`**: Smooth TG curve data containing exactly 1000 points. Ideal for direct import into OriginLab or Python plotting libraries.
    *   *Columns:* `Temperature_C`, `Weight_percent`.
*   **`TG_raw_data_w_inf_endpoint.csv`**: Contains the raw discrete values corresponding to the optimization conversion levels ($\alpha$).
    *   *Columns:* `Alpha`, `Temperature_C`, `Weight_percent`, `Activation_Energy_J_mol`, `Pre_Exponential_Factor_1_min`, `Mechanism_Contribution_Rate`.
*   **`TG_simulation_results_w_inf.mat`**: Complete MATLAB binary workspace structure containing the structured variable `detailed_results`.

### 2. Publication-Quality Figures
*   **`TG_curve_w_inf_endpoint.png`**: The main optimized TG curve plot aligned with the predicted neural network char yield ($w_{\infty}$).
*   **`TG_comparison_plot.png`**: Direct comparison between the optimized multi-mechanism simulation curve and the experimental thermogravimetric data.
*   **`TG_contribution_plot.png`**: **Instantaneous Mechanism Contribution Stacked Area Plot**. Shows the dynamic, temperature-dependent contribution rate of each individual mechanism during devolatilization (crucial for physical interpretation).

---

## II. MATLAB Post-Processing Scripts

### 1. Extract Best Model Parameters
To extract parameters from the structured `detailed_results` array without dimension mismatches:

```matlab
% 1. Load the results workspace
load('TG_simulation_results_w_inf.mat');

% 2. Extract best model information
best_category = detailed_results.best_model_category{1};
best_type     = detailed_results.best_model_type{1};
final_error   = detailed_results.final_error_vs_experimental;

fprintf('=======================================\n');
fprintf('Best Model Category: %s\n', best_category);
fprintf('Best Specific Model: %s\n', best_type);
fprintf('Final Penalty-Weighted Error: %.4f%%\n', final_error);
fprintf('=======================================\n');

% 3. Extract and display best mechanism parameters (e.g. series steps)
best_params = detailed_results.best_model_params{1};
if strcmp(best_category, 'multi')
    fprintf('Multi-Mechanism Mode: %s\n', best_params.mode);
    if isfield(best_params, 'mechanisms')
        for i = 1:numel(best_params.mechanisms)
            mech = best_params.mechanisms{i};
            n_val = best_params.mechParams{i}.n;
            fprintf('  Step %d: %-30s | Exponent n = %.4f\n', i, mech, n_val);
        end
    end
end
```

### 2. Plot Predicted vs. Experimental TG & DTG Curves
To generate custom publication-ready plots comparing the simulated kinetics with experimental data:

```matlab
% Load datasets
exp_data  = readtable('../ExpTGdata.csv');
pred_data = readtable('TG_curve_w_inf_endpoint.csv');

% Calculate DTG curves (mass loss rates)
dT_exp   = max(gradient(exp_data.Temperature), 1e-5);
dtg_exp  = -gradient(exp_data.Weight) ./ dT_exp;

dT_pred  = max(gradient(pred_data.Temperature_C), 1e-5);
dtg_pred = -gradient(pred_data.Weight_percent) ./ dT_pred;

% Create figure with dual y-axes (TG left, DTG right)
figure('Units', 'inches', 'Position', [1, 1, 6.5, 5], 'Color', 'w');
yyaxis left
plot(exp_data.Temperature, exp_data.Weight, 'k-', 'LineWidth', 1.5, 'DisplayName', 'Exp TG');
hold on;
plot(pred_data.Temperature_C, pred_data.Weight_percent, 'r--', 'LineWidth', 2, 'DisplayName', 'Sim TG');
ylabel('Weight (wt.%)', 'FontSize', 12, 'FontWeight', 'bold');
ylim([15, 105]);

yyaxis right
plot(exp_data.Temperature, dtg_exp, 'Color', [0.5, 0.5, 0.5], 'LineStyle', '-', 'LineWidth', 1.2, 'DisplayName', 'Exp DTG');
hold on;
plot(pred_data.Temperature_C, dtg_pred, 'b--', 'LineWidth', 1.5, 'DisplayName', 'Sim DTG');
ylabel('DTG Rate (wt.%/°C)', 'FontSize', 12, 'FontWeight', 'bold');
ylim([-0.05, 0.8]);

xlabel('Temperature (°C)', 'FontSize', 12, 'FontWeight', 'bold');
title('Biomass Pyrolysis: Experimental vs. Kinetics Fit', 'FontSize', 13);
legend('Location', 'northeast');
grid on;
set(gca, 'FontSize', 11, 'LineWidth', 1.2);
```

### 3. Plot Mechanism Contribution Over Temperature
To inspect the stacked area plot representing mechanism contributions (stored inside `detailed_results.contribution`):

```matlab
% Extract contribution table
contrib = detailed_results.contribution;
temps   = contrib.T_C;
rates   = contrib.rates;
labels  = contrib.labels;

% Plot Stacked Area
figure('Units', 'inches', 'Position', [1, 1, 6.5, 4.5], 'Color', 'w');
area(temps, rates * 100);
xlabel('Temperature (°C)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Instantaneous Contribution (%)', 'FontSize', 12, 'FontWeight', 'bold');
title('Kinetics Mechanism Contribution Profile', 'FontSize', 13);
legend(labels, 'Location', 'southwest', 'Interpreter', 'none');
grid on;
set(gca, 'FontSize', 11, 'LineWidth', 1.2);
```

---

## III. Verification & Validation Metrics

Before using these curves for manuscript publication, ensure the following parameters are validated:
1.  **Mass Conservation**: Check that final weight matches neural network predicted residue $w_{\infty}$ exactly (deviation $< 0.01\%$).
2.  **Physical Monotonicity**: Weight curve must be strictly monotonically decreasing. The presence of slope rebounds indicates numerical ODE integration instability.
3.  **Physical Parametric Bounds**: Verify that the optimized exponents $n$ for Jander diffusion or Avrami nucleation do not sit exactly on the bounds ($0.1$ or $10.0$). A value stuck on the bounds indicates local boundary traps, requiring a rerun with adjusted GA initialization.