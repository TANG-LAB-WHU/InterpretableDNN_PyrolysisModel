%% CornStover Kinetic Simulation - MODULAR ARCHITECTURE WITH UNIFIED GA OPTIMIZATION
% Uses neural network predicted char yield (w_inf) as the final endpoint for TG curve
% Implements modular adaptive kinetic model optimization with experimental data validation
% Features unified GA optimization and kinetic integration error calculation
% -------------------------------------------------------------------------

clear; clc;

%% Setup logging with performance optimization
rootDir = fileparts(mfilename("fullpath"));
logDirEnv = getenv('MATLAB_LOG_DIR');
if ~isempty(logDirEnv)
  logDir = logDirEnv;
else
  % Fallback to a stable working directory if env var is not set
  logDir = fullfile(pwd, "Results_MultipleMode");
end
if ~exist(logDir, "dir")
  mkdir(logDir);
end

resultsDirEnv = getenv('MATLAB_RESULTS_DIR');
if ~isempty(resultsDirEnv)
  resultsDir = resultsDirEnv;
else
  % Default result root alongside persistent log root
  resultsDir = fullfile(fileparts(logDir), 'Results-MultiMechanism');
end
plotsDir = fullfile(resultsDir, 'Plots');
if ~exist(resultsDir, "dir")
  mkdir(resultsDir);
end
if ~exist(plotsDir, "dir")
  mkdir(plotsDir);
end

% Create log file with timestamp
timestamp = datestr(now, "yyyy-mm-dd_HH-MM-SS");
logFile = fullfile(logDir, sprintf("cornstover_simulation_%s.log", timestamp));

% Start diary to capture all output
diary(logFile);
diaryCleanup = onCleanup(@() diary('off'));
fprintf("=== CORNSTOVER SIMULATION LOG ===\n");
fprintf("Started: %s\n", datestr(now));
fprintf("Script: %s\n", mfilename("fullpath"));
fprintf("MATLAB Version: %s\n", version);
fprintf("Current Working Directory: %s\n", pwd);
fprintf("Log File: %s\n", logFile);
fprintf("Results Directory: %s\n", resultsDir);
fprintf("Modular architecture with unified GA optimization\n");
fprintf("Kinetic integration error calculation enabled\n");
fprintf("================================\n\n");
tic;

%% Setup and path configuration
rootDir = fileparts(mfilename("fullpath"));

% Add project paths to the END of the search path to avoid shadowing
% MATLAB toolbox internals (e.g., parallel.internal.customattr.CustomPropTypes)
addpath(fullfile(rootDir, '..', 'bpDNN2Ea_AshOptimized'), '-end');
addpath(fullfile(rootDir, '..', 'bpDNN2Yield_AshOptimized'), '-end');
addpath(fullfile(rootDir, '..', 'CallingFunctions'), '-end');
addpath(fullfile(rootDir, '..', 'CallingFunctions', 'AdaptiveKineticModels'), '-end');
addpath(fullfile(rootDir, '..', 'CallingFunctions', 'AdaptiveKineticModels', 'Core'), '-end');
addpath(fullfile(rootDir, '..', 'CallingFunctions', 'AdaptiveKineticModels', 'GA'), '-end');
addpath(fullfile(rootDir, '..', 'CallingFunctions', 'AdaptiveKineticModels', 'Models'), '-end');
addpath(fullfile(rootDir, '..', 'CallingFunctions', 'AdaptiveKineticModels', 'Optimization'), '-end');
addpath(fullfile(rootDir, '..', 'CallingFunctions', 'AdaptiveKineticModels', 'Evaluation'), '-end');
addpath(fullfile(rootDir, '..', 'CallingFunctions', 'AdaptiveKineticModels', 'Utils'), '-end');

% Verify Parallel Computing Toolbox is not shadowed
pctClass = which('parallel.internal.customattr.CustomPropTypes');
if isempty(pctClass)
  warning('Parallel Computing Toolbox class not found — parpool may fail.');
else
  fprintf('PCT path verified: %s\n', fileparts(pctClass));
end

% Print performance mode
try
  cfgPerf = performanceConfig();
  if isfield(cfgPerf, 'fastMode') && cfgPerf.fastMode
    fprintf('Performance mode: FAST (reduced GA/steps, interpolated error)\n');
  else
    fprintf('Performance mode: ACCURACY (full GA/steps, kinetic error)\n');
  end
catch
  % If config not available, continue silently
end

% Check for required functions
if ~exist('lsqisotonic', 'file')
  error('lsqisotonic function not found. Please ensure it is in the current directory.');
end

%% Load trained neural networks
fprintf('=== LOADING NEURAL NETWORKS ===\n');
try
  EaModelStruct = load(fullfile(rootDir, '..', 'bpDNN2Ea_AshOptimized', 'Results_trained.mat'));
  netEa = EaModelStruct.net;
  PS_Ea = EaModelStruct.PS;
  TS_Ea = EaModelStruct.TS;
  fprintf('Ea prediction network loaded successfully\n');

  YieldModelStruct = load(fullfile(rootDir, '..', 'bpDNN2Yield_AshOptimized', 'Results_trained.mat'));
  netYield = YieldModelStruct.net;
  PS_Y = YieldModelStruct.PS;
  TS_Y = YieldModelStruct.TS;
  fprintf('Yield prediction network loaded successfully\n');
catch ME
  error('Failed to load neural networks: %s', ME.message);
end

%% Load experimental data for optimization
fprintf('\n=== LOADING EXPERIMENTAL DATA FOR OPTIMIZATION ===\n');
try
  expData = readtable('ExpTGdata.csv');
  T_exp = expData{:,1};  % Temperature in Celsius
  w_exp = expData{:,2};  % Weight percentage

  fprintf('Experimental data loaded: %d data points\n', length(T_exp));
  fprintf('Temperature range: %.0f - %.0f°C\n', min(T_exp), max(T_exp));
  fprintf('Weight range: %.1f - %.1f%%\n', min(w_exp), max(w_exp));

catch ME
  error('Failed to load experimental data: %s', ME.message);
end

%% Read Excel data with correct method
inputPath = fullfile(rootDir, 'CornStover_China.xlsx');
try
  % Method 1: Try reading with NumHeaderLines only
  try
    inputData = readmatrix(inputPath, 'NumHeaderLines', 1);
    fprintf('Excel data loaded with NumHeaderLines: %d rows, %d columns\n', size(inputData));
  catch
    % Method 2: Read all data and remove header manually
    fprintf('Trying alternative reading method...\n');
    allData = readmatrix(inputPath);
    inputData = allData(2:end, :);  % Skip first row (header)
    fprintf('Excel data loaded manually: %d rows, %d columns\n', size(inputData));
  end

  % Verify column count
  if size(inputData, 2) ~= 259
    error('Expected 259 columns, but got %d columns', size(inputData, 2));
  end

catch ME
  error('Cannot read %s. Error: %s', inputPath, ME.message);
end

% Use first row as base case for Ea network (represents corn stover)
baseRow = inputData(1, :);
fprintf('Base case data for Ea network: %d columns available\n', length(baseRow));

% For yield prediction, use ALL data rows (not just first row)
allDataRows = inputData;  % Use all rows except header for yield prediction
fprintf('All data rows for yield prediction: %d rows, %d columns\n', size(allDataRows));

% Verify data integrity
if size(allDataRows, 2) ~= 259
  error('All data rows should have 259 features, but have %d', size(allDataRows, 2));
end

% Extract key parameters for display
target_temp_celsius = baseRow(20);  % Target temperature from Excel
heating_rate = baseRow(22);         % Heating rate

% Find temperature range from all data
all_target_temps = allDataRows(:, 20);  % All target temperatures
min_temp = min(all_target_temps);
max_temp = max(all_target_temps);

fprintf('\n=== BASE CASE PARAMETERS ===\n');
fprintf('Base case target temperature: %.0f C\n', target_temp_celsius);
fprintf('Temperature range in Excel: %.0f C to %.0f C\n', min_temp, max_temp);
fprintf('Heating rate: %.0f K/min\n', heating_rate);

% Display some key feature values for verification
fprintf('Key features validation:\n');
fprintf('  Location: %.0f\n', baseRow(1));
fprintf('  VolatileMatters: %.2f%%\n', baseRow(2));
fprintf('  FixedCarbon: %.2f%%\n', baseRow(3));
fprintf('  Ash: %.2f%%\n', baseRow(4));
fprintf('  C: %.2f%%\n', baseRow(5));
fprintf('  FeedstockType_1: %.2f\n', baseRow(24));
fprintf('  MixingRatio_1: %.2f\n', baseRow(142));

%% Construct input features
% For Ea network: 256 features
basicFeatures = baseRow(1:19);           % Cols 1-19: basic features
feedstockMixingAll = baseRow(24:259);    % Cols 24-259: all feedstock and mixing features (236 total)

% For Ea network, take only first 236 feedstock+mixing features
expectedFeedstockMixing = 256 - 19 - 1;  % 256 - basic(19) - degree_conversion(1) = 236
feedstockMixingForEa = feedstockMixingAll(1:expectedFeedstockMixing);

fprintf('\n=== FEATURE CONSTRUCTION FOR EA NETWORK ===\n');
fprintf('Basic features (cols 1-19): %d features\n', length(basicFeatures));
fprintf('Feedstock+mixing features (cols 24-259, first 236): %d features\n', length(feedstockMixingForEa));
fprintf('Total for Ea network: %d features (19 + 1 + 236)\n', length(basicFeatures) + 1 + length(feedstockMixingForEa));

% For Yield network: all 259 features for ALL data rows
yieldFeatures = allDataRows;  % Use all rows (not just first row)
numSamples = size(yieldFeatures, 1);

fprintf('\n=== FEATURE CONSTRUCTION FOR YIELD NETWORK ===\n');
fprintf('Using ALL 259 features for ALL %d data rows in yield network\n', numSamples);

% Verify feature dimensions
if size(yieldFeatures, 2) ~= netYield.numInput
  error('Yield input dimension mismatch: constructed %d, network expects %d', size(yieldFeatures, 2), netYield.numInput);
end

%% Define conversion levels
alphaList = [0.01:0.01:0.10, 0.12:0.02:0.30, 0.32:0.01:0.70, 0.72:0.02:0.95, 0.97:0.005:0.99, 0.995, 0.999];
alphaList = unique(alphaList);
na = numel(alphaList);

fprintf('\nUsing %d conversion levels from %.3f to %.3f\n', na, min(alphaList), max(alphaList));

%% Prepare Ea network inputs (256 features with degree_conversion)
fprintf('\n=== PREPARING EA NETWORK INPUTS ===\n');

eaInputMat = zeros(na, 256);
for ii = 1:na
  % Construct: [basic_features(1-19), degree_conversion(20), feedstock_mixing(21-256)]
  eaInputMat(ii, :) = [basicFeatures, alphaList(ii), feedstockMixingForEa];
end

% Verify dimensions
fprintf('Ea input matrix size: %d samples x %d features\n', size(eaInputMat));
if size(eaInputMat, 2) ~= netEa.numInput
  error('Ea input dimension mismatch: constructed %d, network expects %d', size(eaInputMat, 2), netEa.numInput);
end

sampleInpEa = eaInputMat';  % Transpose for nnpredict (features x samples)

%% Prepare Yield network inputs (all 259 features for ALL samples)
fprintf('\n=== PREPARING YIELD NETWORK INPUTS ===\n');

sampleInpY = yieldFeatures';  % Transpose for nnpredict (259 features x numSamples)

fprintf('Yield input dimensions: %d features x %d samples\n', size(sampleInpY, 1), size(sampleInpY, 2));

%% Neural network predictions
fprintf('\n=== NEURAL NETWORK PREDICTIONS ===\n');
global PS TS

% Ea prediction
PS = PS_Ea; TS = TS_Ea;
try
  Ea_pred_kJ = nnpredict(netEa, sampleInpEa);
  Ea_pred = Ea_pred_kJ * 1e3; % Convert to J/mol
  fprintf('Ea prediction successful\n');
  fprintf('  Range: %.1f - %.1f kJ/mol\n', min(Ea_pred_kJ), max(Ea_pred_kJ));
  fprintf('  Mean: %.1f +/- %.1f kJ/mol\n', mean(Ea_pred_kJ), std(Ea_pred_kJ));
catch ME
  error('Ea prediction failed: %s', ME.message);
end

% Yield prediction with ALL 259 features for ALL samples
PS = PS_Y; TS = TS_Y;
try
  yield_pred_all = nnpredict(netYield, sampleInpY);  % Predict for all samples
  fprintf('Yield prediction successful with ALL 259 features for ALL %d samples\n', size(sampleInpY, 2));

  % Find the sample with highest temperature for w_inf determination
  all_temps = allDataRows(:, 20);  % Column 20 contains target temperatures
  [max_temp_value, max_temp_idx] = max(all_temps);

  fprintf('Temperature analysis for w_inf selection:\n');
  fprintf('  Temperature range: %.0f C to %.0f C\n', min(all_temps), max(all_temps));
  fprintf('  Highest temperature: %.0f C (sample %d)\n', max_temp_value, max_temp_idx);

  % Use yield prediction at highest temperature for w_inf
  yield_at_max_temp = yield_pred_all(:, max_temp_idx);
  char_yield_at_max_temp = yield_at_max_temp(1);

  fprintf('  Yield at highest temperature (%.0f C):\n', max_temp_value);
  fprintf('    Char: %.2f%%, Liquid: %.2f%%, Gas: %.2f%% (Sum: %.2f%%)\n', ...
    yield_at_max_temp(1), yield_at_max_temp(2), yield_at_max_temp(3), sum(yield_at_max_temp));

  yield_pred = yield_at_max_temp;  % Use highest temperature prediction for w_inf calculation

catch ME
  error('Yield prediction failed: %s', ME.message);
end

% Validate and normalize yields
charYield_perc = yield_pred(1);
liquidYield_perc = yield_pred(2);
gasYield_perc = yield_pred(3);

% If sum deviates too much from 100%, normalize
if abs(sum(yield_pred) - 100) > 5
  fprintf('  Normalizing yields (sum was %.2f%%)\n', sum(yield_pred));
  yield_pred = yield_pred / sum(yield_pred) * 100;
  charYield_perc = yield_pred(1);
  liquidYield_perc = yield_pred(2);
  gasYield_perc = yield_pred(3);
end

% Apply reasonable bounds for corn stover
charYield_perc = max(15, min(charYield_perc, 35));  % Corn stover char yield typically 15-35%
w_inf = charYield_perc / 100;

fprintf('  Final yields after validation (from highest temperature %.0f C):\n', max_temp_value);
fprintf('    Char: %.2f%% (used as w_inf)\n', charYield_perc);
fprintf('    Liquid: %.2f%%\n', liquidYield_perc);
fprintf('    Gas: %.2f%%\n', gasYield_perc);

%% Enhanced modular adaptive kinetic model optimization
fprintf('\n=== ENHANCED MODULAR ADAPTIVE KINETIC MODEL OPTIMIZATION ===\n');
fprintf('Using modular architecture with the following modules:\n');
fprintf('  - Core: Main controllers and parameter optimization\n');
fprintf('  - GA: Unified genetic algorithm optimization\n');
fprintf('  - Models: Complete kinetic model library\n');
fprintf('  - Optimization: Advanced optimization algorithms\n');
fprintf('  - Evaluation: Kinetic integration error calculation\n');
fprintf('  - Utils: Helper functions and configuration\n');
beta = heating_rate;

% Temperature bounds from Excel data - ALIGNED WITH DATASET
T_start_K = min_temp + 273.15;  % Dataset minimum temperature (e.g., 20°C)
T_end_K = max_temp + 273.15;    % Dataset maximum temperature (e.g., 900°C)

fprintf('Temperature constraints from Excel data (ALIGNED WITH DATASET):\n');
fprintf('  Start: %.0f K (%.0f C) - Dataset minimum\n', T_start_K, T_start_K-273.15);
fprintf('  End: %.0f K (%.0f C) - Dataset maximum\n', T_end_K, T_end_K-273.15);
fprintf('  Range: %.0f K (%.0f C span)\n', T_end_K-T_start_K, (T_end_K-T_start_K));

% Use modular model comparison with experimental data
fprintf('\nStarting comprehensive model comparison with experimental data using modular GA optimization...\n');
fprintf('Using modular architecture with unified GA optimization and kinetic integration error calculation.\n');
fprintf('Single mechanism models use GA optimization only (no parameter combination testing).\n');
[bestModel, allResults] = modelComparison(alphaList, Ea_pred, beta, T_start_K, T_end_K, T_exp, w_exp, w_inf);

% Extract best model results
bestCategory = bestModel.category;
bestParams   = bestModel.params;
bestError    = bestModel.error;

% Robustly determine model label and mechId
if isfield(bestParams, 'modelType')
  bestModelType = bestParams.modelType;
  mechId = sprintf('%s|%s', bestCategory, bestModelType);
else
  % Multi-mechanism or cases without a single modelType
  bestModelType = 'multi';
  mechId = 'multi';
end

fprintf('\n=== OPTIMIZATION RESULTS ===\n');
fprintf('Best model category: %s\n', bestCategory);
fprintf('Best specific model: %s\n', bestModelType);
fprintf('Best model parameters: %s\n', struct2str(bestParams));
fprintf('Best model error vs experimental data: %.2e\n', bestError);

% Generate final TG curve with best model
fprintf('\n=== GENERATING FINAL TG CURVE ===\n');
G_alpha_best = generateGAlpha(alphaList, mechId, bestParams);
[T_opt, A_opt, opt_error] = optimizeTandA(G_alpha_best, Ea_pred, beta, T_start_K, T_end_K, alphaList, T_exp, w_exp, w_inf);

% Generate final TG curve
[w_final, T_final] = generateTGCurveFromG(G_alpha_best, T_opt, A_opt, Ea_pred, beta, alphaList, w_inf, T_exp);

% Calculate final error vs experimental data using kinetic model integration
final_error = calculateErrorWithKinetics(w_final, T_final, w_exp, T_exp, w_inf, G_alpha_best, T_opt, A_opt, Ea_pred, beta, alphaList);

fprintf('Final error vs experimental data: %.4f%%\n', final_error);

%% Generate smooth curves and create plots
T_fine = linspace(min(T_final), max(T_final), 1000);
[T_final_unique, unique_idx] = unique(T_final);
w_final_unique = w_final(unique_idx);
w_fine = interp1(T_final_unique, w_final_unique, T_fine, 'pchip');
w_fine = cummin(w_fine);                    % Maintain monotonicity
w_fine = max(w_fine, w_inf * 100);          % Don't go below char yield
w_fine(end) = w_inf * 100;                  % Ensure exact endpoint

% Create publication-quality TG curve
figure('Visible', 'off', 'Color', 'white', 'Position', [200, 200, 800, 600]);
plot(T_fine, w_fine, 'b-', 'LineWidth', 3); xlim([min(T_exp) max(T_exp)]);
xlabel('Temperature (°C)', 'FontSize', 16, 'FontWeight', 'bold');
ylabel('Weight (%)', 'FontSize', 16, 'FontWeight', 'bold');
title('TG Curve for Corn Stover (w_{inf} from Neural Network)', 'FontSize', 18, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', 14, 'LineWidth', 1.5);

% Add information box
annotation('textbox', [0.65, 0.75, 0.3, 0.2], ...
  'String', sprintf('Char Yield: %.1f%%\nTemp Range: %.0f-%.0f°C\nModel: %s\nEndpoint: w_{inf} NN', ...
  w_inf*100, min(T_fine), max(T_fine), bestModelType), ...
  'FontSize', 12, 'BackgroundColor', 'white', 'EdgeColor', 'black', ...
  'FitBoxToText', 'on');

% Save plot
print(gcf, fullfile(plotsDir, 'TG_curve_w_inf_endpoint.png'), '-dpng', '-r600');

%% Compare with experimental data
fprintf('\n=== COMPARING WITH EXPERIMENTAL DATA ===\n');

% Plot comparison
figure('Visible', 'off', 'Name', 'Optimized vs Experimental TG', 'Position', [100, 100, 800, 600]);
plot(T_exp, w_exp, 'b-', 'LineWidth', 2, 'DisplayName', 'Experimental');
hold on;
w_pred_on_exp = interp1(T_final, w_final, T_exp, 'pchip', w_inf*100);
plot(T_exp, w_pred_on_exp, 'r--', 'LineWidth', 2, 'DisplayName', 'Optimized');
xlabel('Temperature (°C)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Weight (%)', 'FontSize', 12, 'FontWeight', 'bold');
title(sprintf('TG Curve Comparison (Error: %.4f%%)', final_error), 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'best');
grid on;

% Save comparison plot
saveas(gcf, fullfile(plotsDir, 'TG_comparison_plot.png'));

%% Save data
TG_table = table(T_fine', w_fine', 'VariableNames', {'Temperature_C', 'Weight_percent'});
writetable(TG_table, fullfile(resultsDir, 'TG_curve_w_inf_endpoint.csv'));

% Ensure all arrays have the same length for table creation
array_lengths = [length(alphaList), length(T_final), length(w_final), length(Ea_pred_kJ), length(A_opt)];
min_length = min(array_lengths);

% Truncate all arrays to the same length
alphaList_table = alphaList(1:min_length);
T_final_table = T_final(1:min_length);
w_final_table = w_final(1:min_length);
Ea_pred_kJ_table = Ea_pred_kJ(1:min_length);
A_opt_table = A_opt(1:min_length);

% Create table with column vectors
alphaList_col = alphaList_table(:);
T_final_col = T_final_table(:);
w_final_col = w_final_table(:);
Ea_pred_kJ_col = Ea_pred_kJ_table(:);
A_opt_col = A_opt_table(:);

raw_data_table = table(alphaList_col, T_final_col, w_final_col, Ea_pred_kJ_col, A_opt_col, ...
  'VariableNames', {'Alpha', 'Temperature_C', 'Weight_percent', 'Ea_kJ_mol', 'A_preExp'});

% ---------------------------------------------------------------------
% Calculate and append instantaneous mechanism relative contribution rates
% ---------------------------------------------------------------------
fprintf('\n=== CALCULATING INSTANTANEOUS MECHANISM CONTRIBUTION RATES ===\n');
contribution = struct();
try
  contribution = calculateContributionRates(alphaList, mechId, bestParams);
  contrib_rates_table = contribution.rates(1:min_length, :);
  
  for k = 1:numel(contribution.labels)
    colName = matlab.lang.makeValidName(contribution.labels{k});
    raw_data_table.(colName) = contrib_rates_table(:, k);
  end
  fprintf('Contribution rates successfully calculated and appended to table.\n');
catch ME
  fprintf('Warning: failed to calculate contribution rates: %s\n', ME.message);
  contribution.labels = {mechId};
  contribution.rates = ones(length(alphaList), 1);
end

writetable(raw_data_table, fullfile(resultsDir, 'TG_raw_data_w_inf_endpoint.csv'));

% ---------------------------------------------------------------------
% Generate and save beautiful stacked area plot for contribution rates
% ---------------------------------------------------------------------
try
  if numel(contribution.labels) > 1 || ~strcmp(contribution.labels{1}, 'multi')
    figure('Visible', 'off', 'Name', 'Mechanism Contribution Rates', 'Position', [150, 150, 800, 600], 'Color', 'white');
    
    % Plot stacked area
    area(alphaList_table, contrib_rates_table * 100);
    ylim([0 100]);
    xlim([0 1]);
    xlabel('Conversion Level (\alpha)', 'FontSize', 14, 'FontWeight', 'bold');
    ylabel('Mechanism Relative Contribution Rate (%)', 'FontSize', 14, 'FontWeight', 'bold');
    title('Mechanism Contribution Rates across Pyrolysis Progress', 'FontSize', 16, 'FontWeight', 'bold');
    
    % Clean up legend labels
    legendLabels = cell(size(contribution.labels));
    for k = 1:numel(contribution.labels)
      lbl = contribution.labels{k};
      lbl = regexprep(lbl, '^(parallel|series|branch)_\d+_', '');
      legendLabels{k} = strrep(lbl, '_', ' ');
    end
    legend(legendLabels, 'Location', 'eastoutside', 'FontSize', 11);
    grid on;
    set(gca, 'FontSize', 12, 'LineWidth', 1.5);
    
    print(gcf, fullfile(plotsDir, 'TG_contribution_plot.png'), '-dpng', '-r600');
    close(gcf);
    fprintf('Saved beautiful mechanism contribution stacked area plot: TG_contribution_plot.png\n');
  end
catch ME
  fprintf('Warning: failed to generate stacked area plot: %s\n', ME.message);
end

% Save detailed results
detailed_results = struct();
% Use the same truncated arrays for detailed results
detailed_results.alphaList = alphaList_table;
detailed_results.temperatures_C = T_final_table;
detailed_results.weights_percent = w_final_table;
detailed_results.activation_energies_kJ_mol = Ea_pred_kJ_table;
detailed_results.pre_exponential_factors = A_opt_table;
detailed_results.w_inf_from_neural_network = w_inf;
detailed_results.char_yield_percent = w_inf * 100;
detailed_results.best_model_category = bestCategory;
detailed_results.best_model_type = bestModelType;
detailed_results.best_model_params = bestParams;
detailed_results.temperature_range_C = [min(T_final), max(T_final)];
detailed_results.excel_temperature_range_C = [min_temp, max_temp];
detailed_results.final_error_vs_experimental = final_error;
detailed_results.contribution = contribution;

save(fullfile(resultsDir, 'TG_simulation_results_w_inf.mat'), 'detailed_results');

%% Final summary and validation
executionTime = toc;
fprintf('\n=== FINAL SUMMARY ===\n');
fprintf('TG curve constructed with w_inf as endpoint from neural network\n');
fprintf('Neural network predictions:\n');
fprintf('  - Activation energies: %.1f ± %.1f kJ/mol\n', mean(Ea_pred_kJ), std(Ea_pred_kJ));
fprintf('  - Char yield (w_inf): %.2f%% (from highest temperature sample)\n', w_inf * 100);
fprintf('  - Temperature range: %.0f - %.0f°C (within Excel range: %.0f - %.0f°C)\n', ...
  min(T_final), max(T_final), min_temp, max_temp);

fprintf('\nModular GA optimization results:\n');
fprintf('  - Best model category: %s\n', bestCategory);
fprintf('  - Best specific model: %s\n', bestModelType);
fprintf('  - Best model parameters: %s\n', struct2str(bestParams));
fprintf('  - Final error vs experimental data: %.4f%%\n', final_error);
fprintf('  - Using kinetic integration for error calculation\n');
fprintf('  - Unified GA optimization for both single and multi-mechanism modes\n');
fprintf('  - Single mechanism models use GA optimization only (no parameter combination testing)\n');

fprintf('\nData files generated:\n');
fprintf('  - TG_curve_w_inf_endpoint.csv (smooth curve data)\n');
fprintf('  - TG_raw_data_w_inf_endpoint.csv (raw optimization data)\n');
fprintf('  - TG_simulation_results_w_inf.mat (complete results)\n');
fprintf('  - TG_curve_w_inf_endpoint.png (main TG curve plot)\n');
fprintf('  - TG_comparison_plot.png (experimental comparison)\n');

% Validation checks
temp_valid = (min(T_final) >= min_temp) && (max(T_final) <= max_temp);
yield_valid = (w_inf*100 >= 10) && (w_inf*100 <= 40);
monotonic_valid = all(diff(w_final) <= 0);
endpoint_valid = abs(w_final(end) - w_inf * 100) < 0.01;

fprintf('\n=== VALIDATION CHECKS ===\n');
fprintf('Temperature range within Excel bounds: %s\n', string(temp_valid));
fprintf('Char yield reasonable (10-40%%): %s\n', string(yield_valid));
fprintf('Weight monotonically decreasing: %s\n', string(monotonic_valid));
fprintf('Endpoint matches neural network w_inf: %s\n', string(endpoint_valid));

if temp_valid && yield_valid && monotonic_valid && endpoint_valid
  fprintf('\n✓ ALL VALIDATION CHECKS PASSED\n');
  fprintf('✓ TG curve successfully generated with w_inf as endpoint\n');
  fprintf('✓ Neural network char yield (%.2f%%) used as final residue\n', w_inf * 100);
  fprintf('✓ Modular architecture optimization completed successfully\n');
else
  fprintf('\n⚠ Some validation checks failed - review results\n');
end

fprintf('\nExecution time: %.2f seconds\n', executionTime);
fprintf('Results saved to: %s\n', resultsDir);

fprintf('\n=== SIMULATION COMPLETE ===\n');
diary off;
clear diaryCleanup;

% Utility function
function str = struct2str(s)
% Convert struct to string representation
if ~isstruct(s)
  str = sprintf('Non-struct value: %s', mat2str(s));
elseif isempty(fieldnames(s))
  str = '{}';
else
  fields = fieldnames(s);
  values = struct2cell(s);

  % Build key:value pairs with robust type handling
  strParts = cell(1, numel(fields));
  for i = 1:numel(fields)
    val = values{i};
    if isnumeric(val)
      valStr = num2str(val);
    elseif ischar(val)
      valStr = val;
    elseif islogical(val)
      valStr = mat2str(val);
    elseif iscell(val)
      valStr = sprintf('cell[%d]', numel(val));
    elseif isstruct(val)
      valStr = struct2str(val);   % recursive handling for nested structs
    else
      valStr = '<non-displayable>';
    end
    strParts{i} = [fields{i}, ':', valStr];
  end
  str = ['{', strjoin(strParts, ', '), '}'];
end
end

function contribution = calculateContributionRates(alphaList, mechId, params)
% Calculate instantaneous relative contribution rates eta_k(alpha) of mechanisms across pyrolysis progress
%
% Input:
%   alphaList - Conversion level list
%   mechId - Best mechanism category/ID (e.g. 'multi' or 'diffusion|jander_3d')
%   params - Best model parameters struct
%
% Output:
%   contribution - struct containing:
%     .labels - cell array of names/identifiers of the components
%     .rates  - matrix [length(alphaList) x numel(labels)] of contribution rates (0 to 1)

alphaList = alphaList(:);
N = length(alphaList);
contribution = struct();

if strcmp(mechId, 'multi')
  mode = params.mode;
  if strcmp(mode, 'parallel')
    numMechs = numel(params.mechanisms);
    rates = zeros(N, numMechs);
    labels = cell(1, numMechs);
    
    for k = 1:numMechs
      subId = params.mechanisms{k};
      subParams = params.mechParams{k};
      mechG = generateGAlpha(alphaList, subId, subParams);
      
      % Compute numerical derivative with respect to alpha
      dG = gradient(mechG, alphaList);
      dG = max(0, dG); % Ensure non-negative
      
      rates(:, k) = params.weights(k) * dG;
      labels{k} = sprintf('parallel_%d_%s', k, subId);
    end
    
    % Normalize to sum to 1 at each alpha
    sum_rates = sum(rates, 2);
    sum_rates(sum_rates == 0) = 1;
    rates = rates ./ sum_rates;
    
    contribution.labels = labels;
    contribution.rates = rates;
    
  elseif strcmp(mode, 'hybrid')
    numBranches = numel(params.branches);
    rates = zeros(N, numBranches);
    labels = cell(1, numBranches);
    
    for b = 1:numBranches
      branch = params.branches{b};
      branchG = alphaList;
      for m = 1:numel(branch.mechanisms)
        subId = branch.mechanisms{m};
        subParams = branch.mechParams{m};
        branchG = generateGAlpha(branchG, subId, subParams);
      end
      
      dG = gradient(branchG, alphaList);
      dG = max(0, dG);
      
      rates(:, b) = params.weights(b) * dG;
      
      % Build label from sub-mechanisms in the branch
      sub_labels = branch.mechanisms;
      labels{b} = sprintf('branch_%d_(%s)', b, strjoin(sub_labels, '_seq_'));
    end
    
    sum_rates = sum(rates, 2);
    sum_rates(sum_rates == 0) = 1;
    rates = rates ./ sum_rates;
    
    contribution.labels = labels;
    contribution.rates = rates;
    
  elseif strcmp(mode, 'series')
    numMechs = numel(params.mechanisms);
    rates = zeros(N, numMechs);
    labels = cell(1, numMechs);
    
    for k = 1:numMechs
      subId = params.mechanisms{k};
      subParams = params.mechParams{k};
      mechG = generateGAlpha(alphaList, subId, subParams);
      dG = gradient(mechG, alphaList);
      dG = max(0, dG);
      rates(:, k) = dG;
      labels{k} = sprintf('series_%d_%s', k, subId);
    end
    
    sum_rates = sum(rates, 2);
    sum_rates(sum_rates == 0) = 1;
    rates = rates ./ sum_rates;
    
    contribution.labels = labels;
    contribution.rates = rates;
  end
else
  % Single model: 100% contribution
  contribution.labels = {mechId};
  contribution.rates = ones(N, 1);
end
end