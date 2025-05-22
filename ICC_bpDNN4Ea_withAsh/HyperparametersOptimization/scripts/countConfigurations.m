function countConfigurations()
% COUNTCONFIGURATIONS Calculate the total number of hyperparameter combinations and print information

% Load hyperparameter configuration
script_dir = fileparts(mfilename('fullpath'));
config_file = fullfile(script_dir, '..', 'hyperparameters.json');
fid = fopen(config_file, 'r');
raw_text = fread(fid, '*char')';
fclose(fid);
config = jsondecode(raw_text);

% Calculate all possible combinations
num_lr = length(config.learning_rate);
num_momentum = length(config.momentum);
num_hidden_layers = length(config.hidden_layers);
num_transfer_funcs = length(config.transfer_functions);
num_lr_inc = length(config.lr_increase_factor);
num_lr_dec = length(config.lr_decrease_factor);
num_div_ratio = size(config.division_ratio, 1);
num_max_fail = length(config.max_fail);
num_max_epochs = length(config.max_epochs);

total_combinations = num_lr * num_momentum * num_hidden_layers * ...
                     num_transfer_funcs * num_lr_inc * num_lr_dec * ...
                     num_div_ratio * num_max_fail * num_max_epochs;

% Print information
fprintf('Hyperparameter Combination Statistics:\n');
fprintf('------------------------------------------------\n');
fprintf('Learning Rate: %d options\n', num_lr);
fprintf('Momentum Coefficient: %d options\n', num_momentum);
fprintf('Hidden Layer Structures: %d configurations\n', num_hidden_layers);
fprintf('Transfer Function Combinations: %d configurations\n', num_transfer_funcs);
fprintf('Learning Rate Increase Factor: %d options\n', num_lr_inc);
fprintf('Learning Rate Decrease Factor: %d options\n', num_lr_dec);
fprintf('Data Division Ratios: %d configurations\n', num_div_ratio);
fprintf('Maximum Validation Failures: %d options\n', num_max_fail);
fprintf('Maximum Training Epochs: %d options\n', num_max_epochs);
fprintf('------------------------------------------------\n');
fprintf('Total combinations: %d\n', total_combinations);

% Estimate computation time
avg_time_per_config = 60; % Assume average training time of 60 seconds per configuration
total_time_seconds = total_combinations * avg_time_per_config;
total_time_hours = total_time_seconds / 3600;
total_time_days = total_time_hours / 24;

fprintf('Estimated total computation time (single core):\n');
fprintf('  %.2f hours\n', total_time_hours);
fprintf('  %.2f days\n', total_time_days);

% Estimate parallel computation time
for num_nodes = [8, 16, 32, 64, 128, 256]
    parallel_time_hours = total_time_hours / num_nodes;
    parallel_time_days = parallel_time_hours / 24;
    fprintf('Estimated time using %d compute nodes:\n', num_nodes);
    fprintf('  %.2f hours\n', parallel_time_hours);
    fprintf('  %.2f days\n', parallel_time_days);
end
end
