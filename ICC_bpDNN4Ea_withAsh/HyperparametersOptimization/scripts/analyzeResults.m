function analyzeResults(results_dir)
% ANALYZERESULTS Analyze the results of hyperparameter optimization
% Input parameters:
%   results_dir: Directory containing all configuration results

% Find all configuration folders
config_dirs = dir(fullfile(results_dir, 'config_*'));
config_dirs = config_dirs([config_dirs.isdir]);

% Initialize results array
results = struct('config_id', {}, 'train_mse', {}, 'val_mse', {}, 'test_mse', {}, ...
    'train_r2', {}, 'val_r2', {}, 'test_r2', {}, 'epochs', {}, 'training_time', {}, ...
    'learning_rate', {}, 'momentum', {}, 'hidden_layers', {}, 'transfer_functions', {}, ...
    'lr_increase_factor', {}, 'lr_decrease_factor', {}, 'division_ratio', {}, ...
    'max_fail', {}, 'max_epochs', {}, 'stop_reason', {});

% Collect all results
for i = 1:length(config_dirs)
    config_dir = fullfile(results_dir, config_dirs(i).name);
    perf_file = fullfile(config_dir, 'performance_summary.csv');
    config_file = fullfile(config_dir, 'config_info.csv');
    
    if exist(perf_file, 'file') && exist(config_file, 'file')
        % Read performance summary
        perf_data = readtable(perf_file);
        config_data = readtable(config_file);
        
        % Merge data
        results(i).config_id = perf_data.config_id;
        results(i).train_mse = perf_data.train_mse;
        results(i).val_mse = perf_data.val_mse;
        results(i).test_mse = perf_data.test_mse;
        results(i).train_r2 = perf_data.train_r2;
        results(i).val_r2 = perf_data.val_r2;
        results(i).test_r2 = perf_data.test_r2;
        results(i).epochs = perf_data.epochs;
        results(i).training_time = perf_data.training_time;
        results(i).stop_reason = string(perf_data.stop_reason);
        
        % Add configuration information
        results(i).learning_rate = config_data.learning_rate;        results(i).momentum = config_data.momentum;
        results(i).hidden_layers = string(config_data.hidden_layers);
        results(i).transfer_functions = string(config_data.transfer_functions);
        results(i).lr_increase_factor = config_data.lr_increase_factor;
        results(i).lr_decrease_factor = config_data.lr_decrease_factor;
        results(i).division_ratio = string(config_data.division_ratio);
        results(i).max_fail = config_data.max_fail;
        results(i).max_epochs = config_data.max_epochs;
    end
end

% Convert to table
results_table = struct2table(results);

% Sort by validation MSE
[~, sort_idx] = sort([results.val_mse]);
sorted_results = results_table(sort_idx, :);

% Save sorted results
writetable(sorted_results, fullfile(results_dir, 'all_results_sorted.csv'));

% Find best configuration
best_idx = find([results.val_mse] == min([results.val_mse]), 1);
best_config = results(best_idx);

% Save best configuration information
fid = fopen(fullfile(results_dir, 'best_config.txt'), 'w');
fprintf(fid, 'Best Configuration (ID: %d)\n', best_config.config_id);
fprintf(fid, '----------------------------------------\n');
fprintf(fid, 'Performance Metrics:\n');
fprintf(fid, '  Training MSE: %.6f\n', best_config.train_mse);
fprintf(fid, '  Validation MSE: %.6f\n', best_config.val_mse);
fprintf(fid, '  Testing MSE: %.6f\n', best_config.test_mse);
fprintf(fid, '  Training R^2: %.4f\n', best_config.train_r2);
fprintf(fid, '  Validation R^2: %.4f\n', best_config.val_r2);
fprintf(fid, '  Testing R^2: %.4f\n', best_config.test_r2);
fprintf(fid, '  Training Epochs: %d\n', best_config.epochs);
fprintf(fid, '  Training Time: %.2f seconds\n', best_config.training_time);
fprintf(fid, '  Stop Reason: %s\n', best_config.stop_reason);
fprintf(fid, '----------------------------------------\n');
fprintf(fid, 'Hyperparameters:\n');
fprintf(fid, '  Learning Rate: %.4f\n', best_config.learning_rate);
fprintf(fid, '  Momentum Coefficient: %.4f\n', best_config.momentum);
fprintf(fid, '  Hidden Layer Structure: %s\n', best_config.hidden_layers);
fprintf(fid, '  Transfer Functions: %s\n', best_config.transfer_functions);
fprintf(fid, '  Learning Rate Increase Factor: %.4f\n', best_config.lr_increase_factor);
fprintf(fid, '  Learning Rate Decrease Factor: %.4f\n', best_config.lr_decrease_factor);
fprintf(fid, '  Data Division Ratio: %s\n', best_config.division_ratio);
fprintf(fid, '  Maximum Validation Failures: %d\n', best_config.max_fail);
fprintf(fid, '  Maximum Training Epochs: %d\n', best_config.max_epochs);
fclose(fid);

% Create performance visualization
createPerformanceVisualization(results, results_dir);

fprintf('Analysis completed. Best configuration ID: %d, Validation MSE: %.6f\n', best_config.config_id, best_config.val_mse);
end

function createPerformanceVisualization(results, output_dir)
% Create performance visualization charts

% Extract data
learning_rates = [results.learning_rate];
momentums = [results.momentum];
val_mses = [results.val_mse];
test_mses = [results.test_mse];
train_r2s = [results.train_r2];
val_r2s = [results.val_r2];
test_r2s = [results.test_r2];
training_times = [results.training_time];

% Learning Rate vs Validation MSE relationship
figure('visible', 'off');
scatter(learning_rates, val_mses, 50, momentums, 'filled');
colorbar;
colormap('jet');
title('Learning Rate vs Validation MSE (color indicates momentum)');
xlabel('Learning Rate');
ylabel('Validation MSE');
grid on;
saveas(gcf, fullfile(output_dir, 'lr_vs_val_mse.fig'));
saveas(gcf, fullfile(output_dir, 'lr_vs_val_mse.png'));

% Momentum vs Validation MSE relationship
figure('visible', 'off');
scatter(momentums, val_mses, 50, learning_rates, 'filled');
colorbar;
colormap('jet');
title('Momentum vs Validation MSE (color indicates learning rate)');
xlabel('Momentum');
ylabel('Validation MSE');
grid on;
saveas(gcf, fullfile(output_dir, 'momentum_vs_val_mse.fig'));
saveas(gcf, fullfile(output_dir, 'momentum_vs_val_mse.png'));

% Training Time vs Validation MSE relationship
figure('visible', 'off');
scatter(training_times, val_mses, 50, 'filled');
title('Training Time vs Validation MSE');
xlabel('Training Time (seconds)');
ylabel('Validation MSE');
grid on;
saveas(gcf, fullfile(output_dir, 'time_vs_val_mse.fig'));
saveas(gcf, fullfile(output_dir, 'time_vs_val_mse.png'));

% R^2 comparison
figure('visible', 'off');
boxplot([train_r2s', val_r2s', test_r2s'], 'Labels', {'Training R^2', 'Validation R^2', 'Testing R^2'});
title('R^2 Distribution Comparison');
grid on;
saveas(gcf, fullfile(output_dir, 'r2_comparison.fig'));
saveas(gcf, fullfile(output_dir, 'r2_comparison.png'));

% Top 10 best configurations performance comparison
[~, sort_idx] = sort(val_mses);
top10_idx = sort_idx(1:min(10, length(sort_idx)));
config_ids = [results(top10_idx).config_id];
top10_val_mses = val_mses(top10_idx);
top10_test_mses = test_mses(top10_idx);

figure('visible', 'off');
bar([top10_val_mses', top10_test_mses']);
title('Performance Comparison of Top 10 Configurations');
xlabel('Configuration ID');
ylabel('MSE');
legend('Validation MSE', 'Test MSE');
set(gca, 'XTickLabel', config_ids);
grid on;
saveas(gcf, fullfile(output_dir, 'top10_performance.fig'));
saveas(gcf, fullfile(output_dir, 'top10_performance.png'));
end
