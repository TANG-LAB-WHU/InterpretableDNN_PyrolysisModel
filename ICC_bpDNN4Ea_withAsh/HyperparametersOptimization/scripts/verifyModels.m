function verifyModels(results_dir, baseline_dir)
% VERIFYMODELS Compare optimized models with baseline models
% Input parameters:
%   results_dir: Directory containing optimization results
%   baseline_dir: Directory containing baseline model and results

fprintf('Verifying optimized models against baseline model...\n');

% Load baseline model results
baseline_file = fullfile(baseline_dir, 'baseline_results.mat');
if ~exist(baseline_file, 'file')
    error('Baseline results file not found: %s', baseline_file);
end

baseline = load(baseline_file);
fprintf('Baseline model metrics - MSE: %.6f, R^2: %.4f\n', ...
    baseline.test_error, baseline.test_r2);

% Load best model information
best_config_file = fullfile(results_dir, 'best_config.txt');
if ~exist(best_config_file, 'file')
    error('Best configuration file not found: %s. Run analyzeResults.m first.', best_config_file);
end

% Extract best configuration ID
fid = fopen(best_config_file, 'r');
content = fread(fid, '*char')';
fclose(fid);
matches = regexp(content, 'Best Configuration \(ID: (\d+)\)', 'tokens');
if isempty(matches)
    error('Could not extract best configuration ID from %s', best_config_file);
end
best_config_id = str2double(matches{1}{1});

% Load best model results
best_model_dir = fullfile(results_dir, sprintf('config_%d', best_config_id));
best_model_file = fullfile(best_model_dir, 'Results_trained.mat');
if ~exist(best_model_file, 'file')
    error('Best model results file not found: %s', best_model_file);
end

best_model = load(best_model_file);
fprintf('Best model metrics - MSE: %.6f, R^2: %.4f\n', ...
    best_model.test_error, best_model.test_r2);

% Calculate improvement percentages
mse_improvement = ((baseline.test_error - best_model.test_error) / baseline.test_error) * 100;
r2_improvement = ((best_model.test_r2 - baseline.test_r2) / baseline.test_r2) * 100;

fprintf('Performance Improvement:\n');
fprintf('  MSE: %.2f%% reduction\n', mse_improvement);
fprintf('  R^2: %.2f%% improvement\n', r2_improvement);

% Generate comparison plots
generateComparisonPlots(baseline, best_model, results_dir);

fprintf('Model verification completed.\n');
end

function generateComparisonPlots(baseline, best_model, output_dir)
% Generate plots comparing baseline and best model
comparison_dir = fullfile(output_dir, 'model_comparison');
if ~exist(comparison_dir, 'dir')
    mkdir(comparison_dir);
end

% Compare training curves
figure('visible', 'off');
semilogy(baseline.tr.epoch, baseline.tr.perf, 'b-', 'LineWidth', 2);
hold on;
semilogy(best_model.tr.epoch, best_model.tr.perf, 'r-', 'LineWidth', 2);
title('Training Performance Comparison');
xlabel('Epochs');
ylabel('MSE (log scale)');
legend('Baseline Model', 'Optimized Model');
grid on;
saveas(gcf, fullfile(comparison_dir, 'Training_Comparison.fig'));
saveas(gcf, fullfile(comparison_dir, 'Training_Comparison.png'));
saveas(gcf, fullfile(comparison_dir, 'Training_Comparison.tif'), 'tiffn');

% Compare test set predictions
figure('visible', 'off');
subplot(2, 1, 1);
plot(baseline.test_targets, baseline.test_output, 'bo');
hold on;
plot([min(baseline.test_targets), max(baseline.test_targets)], ...
     [min(baseline.test_targets), max(baseline.test_targets)], 'k-', 'LineWidth', 1);
title('Baseline Model Predictions vs. Actual Values');
xlabel('Actual Values');
ylabel('Predicted Values');
grid on;

subplot(2, 1, 2);
plot(best_model.test_targets, best_model.test_output, 'ro');
hold on;
plot([min(best_model.test_targets), max(best_model.test_targets)], ...
     [min(best_model.test_targets), max(best_model.test_targets)], 'k-', 'LineWidth', 1);
title('Optimized Model Predictions vs. Actual Values');
xlabel('Actual Values');
ylabel('Predicted Values');
grid on;

saveas(gcf, fullfile(comparison_dir, 'Prediction_Comparison.fig'));
saveas(gcf, fullfile(comparison_dir, 'Prediction_Comparison.png'));
saveas(gcf, fullfile(comparison_dir, 'Prediction_Comparison.tif'), 'tiffn');

% Error distribution comparison
figure('visible', 'off');
baseline_errors = baseline.test_targets - baseline.test_output;
best_model_errors = best_model.test_targets - best_model.test_output;

subplot(2, 1, 1);
histogram(baseline_errors, 30);
title('Baseline Model Error Distribution');
xlabel('Error');
ylabel('Frequency');
grid on;

subplot(2, 1, 2);
histogram(best_model_errors, 30);
title('Optimized Model Error Distribution');
xlabel('Error');
ylabel('Frequency');
grid on;

saveas(gcf, fullfile(comparison_dir, 'Error_Distribution.fig'));
saveas(gcf, fullfile(comparison_dir, 'Error_Distribution.png'));
saveas(gcf, fullfile(comparison_dir, 'Error_Distribution.tif'), 'tiffn');

% Close all figures
close all;

% Save comparison data
comparison = struct();
comparison.baseline_mse = baseline.test_error;
comparison.baseline_r2 = baseline.test_r2;
comparison.best_model_mse = best_model.test_error;
comparison.best_model_r2 = best_model.test_r2;
comparison.mse_improvement_percent = ((baseline.test_error - best_model.test_error) / baseline.test_error) * 100;
comparison.r2_improvement_percent = ((best_model.test_r2 - baseline.test_r2) / baseline.test_r2) * 100;

save(fullfile(comparison_dir, 'model_comparison_results.mat'), '-struct', 'comparison');

% Create comparison summary text file
fid = fopen(fullfile(comparison_dir, 'comparison_summary.txt'), 'w');
fprintf(fid, 'Model Performance Comparison\n');
fprintf(fid, '----------------------------------------\n');
fprintf(fid, 'Baseline Model:\n');
fprintf(fid, '  Test MSE: %.6f\n', baseline.test_error);
fprintf(fid, '  Test R^2: %.4f\n', baseline.test_r2);
fprintf(fid, '  Training Epochs: %d\n', length(baseline.tr.epoch));
fprintf(fid, '\n');
fprintf(fid, 'Optimized Model:\n');
fprintf(fid, '  Test MSE: %.6f\n', best_model.test_error);
fprintf(fid, '  Test R^2: %.4f\n', best_model.test_r2);
fprintf(fid, '  Training Epochs: %d\n', length(best_model.tr.epoch));
fprintf(fid, '\n');
fprintf(fid, 'Performance Improvement:\n');
fprintf(fid, '  MSE: %.2f%% reduction\n', comparison.mse_improvement_percent);
fprintf(fid, '  R^2: %.2f%% improvement\n', comparison.r2_improvement_percent);
fclose(fid);
end
