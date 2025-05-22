function setEarlyStoppingThreshold(results_dir)
% SETEARLYSTOPPINGTHRESHOLD Calculate and set MSE threshold for early stopping
% Input parameters:
%   results_dir: Directory containing optimization results

fprintf('Setting early stopping threshold based on completed configurations...\n');

% Find all configuration folders
config_dirs = dir(fullfile(results_dir, 'config_*'));
config_dirs = config_dirs([config_dirs.isdir]);

if length(config_dirs) < 5
    fprintf('Not enough completed configurations (need at least 5, found %d)\n', length(config_dirs));
    fprintf('Will not set early stopping threshold yet.\n');
    return;
end

% Collect validation MSE values
val_mses = [];
for i = 1:length(config_dirs)
    config_dir = fullfile(results_dir, config_dirs(i).name);
    perf_file = fullfile(config_dir, 'performance_summary.csv');
    
    if exist(perf_file, 'file')
        % Read performance summary
        perf_data = readtable(perf_file);
        val_mses(end+1) = perf_data.val_mse;
    end
end

% Calculate threshold as 1.5x the median validation MSE
% This will allow early stopping of clearly underperforming configurations
if ~isempty(val_mses)
    sorted_mses = sort(val_mses);
    median_mse = sorted_mses(ceil(length(sorted_mses) / 2));
    threshold = median_mse * 1.5;
    
    % Save threshold
    early_stop_file = fullfile(fileparts(results_dir), 'early_stop_threshold.mat');
    save(early_stop_file, 'threshold');
    
    fprintf('Early stopping threshold set to %.6f (1.5x median MSE)\n', threshold);
    fprintf('Configurations with validation MSE above this threshold will be stopped early\n');
else
    fprintf('No validation MSE values found. Cannot set early stopping threshold.\n');
end

end
