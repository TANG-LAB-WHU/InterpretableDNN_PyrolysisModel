% Generate Hyperparameter Configurations
% This script loads the hyperparameters.json file and creates individual configuration files

% Load hyperparameter configuration
fprintf('Loading hyperparameter configuration...\n');
config_file = fullfile(pwd, 'hyperparameters.json');
fid = fopen(config_file, 'r');
raw_text = fread(fid, '*char')';
fclose(fid);
config = jsondecode(raw_text);

% Create configurations directory
configs_dir = fullfile(pwd, 'configs');
if ~exist(configs_dir, 'dir')
    mkdir(configs_dir);
    fprintf('Created configs directory.\n');
end

% Generate all possible combinations
learning_rates = config.learning_rate;
momentums = config.momentum;
hidden_layers_configs = config.hidden_layers;
transfer_functions_configs = config.transfer_functions;
lr_increase_factors = config.lr_increase_factor;
lr_decrease_factors = config.lr_decrease_factor;
division_ratios = config.division_ratio;
max_fails = config.max_fail;
max_epochs_values = config.max_epochs;

% Calculate total combinations
total_combinations = length(learning_rates) * ...
                     length(momentums) * ...
                     length(hidden_layers_configs) * ...
                     length(transfer_functions_configs) * ...
                     length(lr_increase_factors) * ...
                     length(lr_decrease_factors) * ...
                     size(division_ratios, 1) * ...
                     length(max_fails) * ...
                     length(max_epochs_values);

fprintf('Total number of combinations: %d\n', total_combinations);

% Generate parameter grid
fprintf('Generating parameter grid...\n');
param_grid = cell(total_combinations, 1);
idx = 1;

% Define progress tracking variables
total_iter = total_combinations;
progress_step = max(1, floor(total_iter / 100)); % Update progress every 1%
next_progress = progress_step;

% Generate all possible combinations
for lr_idx = 1:length(learning_rates)
    for mc_idx = 1:length(momentums)
        for hl_idx = 1:length(hidden_layers_configs)
            for tf_idx = 1:length(transfer_functions_configs)
                for lr_inc_idx = 1:length(lr_increase_factors)
                    for lr_dec_idx = 1:length(lr_decrease_factors)
                        for div_idx = 1:size(division_ratios, 1)
                            for mf_idx = 1:length(max_fails)
                                for me_idx = 1:length(max_epochs_values)
                                    % Create configuration
                                    config_struct = struct();
                                    config_struct.learning_rate = learning_rates(lr_idx);
                                    config_struct.momentum = momentums(mc_idx);
                                    config_struct.hidden_layers = hidden_layers_configs{hl_idx};
                                    config_struct.transfer_functions = transfer_functions_configs{tf_idx};
                                    config_struct.lr_increase_factor = lr_increase_factors(lr_inc_idx);
                                    config_struct.lr_decrease_factor = lr_decrease_factors(lr_dec_idx);
                                    config_struct.division_ratio = division_ratios(div_idx, :);
                                    config_struct.max_fail = max_fails(mf_idx);
                                    config_struct.max_epochs = max_epochs_values(me_idx);
                                    
                                    % Store configuration
                                    param_grid{idx} = config_struct;
                                    
                                    % Save individual configuration file
                                    config_file = fullfile(configs_dir, sprintf('config_%d.json', idx));
                                    fid = fopen(config_file, 'w');
                                    fprintf(fid, '%s', jsonencode(config_struct, 'PrettyPrint', true));
                                    fclose(fid);
                                    
                                    % Update progress
                                    if idx >= next_progress
                                        fprintf('Progress: %.1f%% (%d/%d configurations)\n', ...
                                            100 * idx / total_iter, idx, total_iter);
                                        next_progress = min(total_iter, next_progress + progress_step);
                                    end
                                    
                                    idx = idx + 1;
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end

% Save full parameter grid
fprintf('Saving complete parameter grid...\n');
save(fullfile(configs_dir, 'param_grid.mat'), 'param_grid');

% Create configuration index file
fprintf('Creating configuration index file...\n');
fid = fopen(fullfile(configs_dir, 'config_index.csv'), 'w');
fprintf(fid, 'config_id,learning_rate,momentum,hidden_layers,transfer_functions,lr_increase_factor,lr_decrease_factor,division_ratio,max_fail,max_epochs\n');

for i = 1:length(param_grid)
    config = param_grid{i};
    
    % Format hidden layers and transfer functions for CSV
    hidden_layers_str = sprintf('%d,', config.hidden_layers);
    hidden_layers_str = ['[' hidden_layers_str(1:end-1) ']'];
    
    transfer_functions_str = '';
    for j = 1:length(config.transfer_functions)
        transfer_functions_str = [transfer_functions_str '"' config.transfer_functions{j} '",'];
    end
    transfer_functions_str = ['[' transfer_functions_str(1:end-1) ']'];
    
    division_ratio_str = sprintf('%.2f,', config.division_ratio);
    division_ratio_str = ['[' division_ratio_str(1:end-1) ']'];
    
    fprintf(fid, '%d,%.4f,%.4f,%s,%s,%.4f,%.4f,%s,%d,%d\n', ...
        i, config.learning_rate, config.momentum, hidden_layers_str, ...
        transfer_functions_str, config.lr_increase_factor, ...
        config.lr_decrease_factor, division_ratio_str, ...
        config.max_fail, config.max_epochs);
end

fclose(fid);

fprintf('Configuration generation completed.\n');
