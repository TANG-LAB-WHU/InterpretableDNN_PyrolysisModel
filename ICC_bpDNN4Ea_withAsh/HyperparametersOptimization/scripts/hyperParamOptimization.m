function hyperParamOptimization(config_id, output_dir)
% HYPERPARAMOPTIMIZATION Hyperparameter optimization function for training BP-DNN model
% Input parameters:
%   config_id: Configuration ID used to select specific parameter combination from grid search
%   output_dir: Output directory for saving results

fprintf('Starting hyperparameter optimization for configuration %d\n', config_id);

% Load configuration
configs_dir = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'configs');
config_file = fullfile(configs_dir, sprintf('config_%d.json', config_id));

if ~exist(config_file, 'file')
    error('Configuration file %s not found. Run generate_configurations.m first.', config_file);
end

% Read configuration
fid = fopen(config_file, 'r');
raw_text = fread(fid, '*char')';
fclose(fid);
current_config = jsondecode(raw_text);

% Load prepared data
data_dir = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'data');
data_file = fullfile(data_dir, 'prepared_data.mat');

if ~exist(data_file, 'file')
    error('Prepared data file %s not found. Run prepare_data.m first.', data_file);
end

load(data_file);
% Create results directory for this configuration
result_dir = fullfile(output_dir, sprintf('config_%d', config_id));
if ~exist(result_dir, 'dir')
    mkdir(result_dir);
end
figure_dir = fullfile(result_dir, 'Figures');
if ~exist(figure_dir, 'dir')
    mkdir(figure_dir);
end

% Save current configuration information
config_info = struct2table(current_config);
writetable(config_info, fullfile(result_dir, 'config_info.csv'));

% Configure network parameters
hiddenLayer = current_config.hidden_layers;
net = nncreate(hiddenLayer);

% Set training parameters
net.trainParam.mc = current_config.momentum;  % Momentum coefficient
net.trainParam.lr = current_config.learning_rate;  % Learning rate
net.trainParam.lr_inc = current_config.lr_increase_factor;  % Learning rate increase factor
net.trainParam.lr_dec = current_config.lr_decrease_factor;  % Learning rate decrease factor
net.trainParam.max_fail = current_config.max_fail;  % Maximum validation failures
net.trainParam.epoch = current_config.max_epochs;  % Maximum training epochs
net.trainParam.goal = 1e-6;
net.trainParam.min_grad = 1e-5;

% Set transfer functions
for i = 1:length(hiddenLayer)
    net.layer{i}.transferFcn = current_config.transfer_functions{i};
end
net.layer{end}.transferFcn = current_config.transfer_functions{end};

% Set normalization
net.processFcn = 'mapminmax';
net.processParam.ymax = 1;
net.processParam.ymin = -1;

% Set data division ratio
net.divideFcn = 'dividerand';
net.divideParam.trainRatio = current_config.division_ratio(1);
net.divideParam.valRatio = current_config.division_ratio(2);
net.divideParam.testRatio = current_config.division_ratio(3);

% Prepare input and target data
XP = Variables;  % Input data
YP = Ea;  % Target data

% Train network
fprintf('Starting training for configuration %d...\n', config_id);
tic;  % Record start time

% Check if early stopping checkpoint file exists
early_stop_file = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'early_stop_threshold.mat');
if exist(early_stop_file, 'file')
    % Load early stopping threshold
    early_stop_data = load(early_stop_file);
    early_stop_threshold = early_stop_data.threshold;
    fprintf('Using early stopping with MSE threshold: %.6f\n', early_stop_threshold);
    
    % Custom training function with early stopping
    [net, tr] = trainWithEarlyStopping(net, XP', YP', early_stop_threshold);
else
    % Regular training without early stopping
    [net, tr] = nntrain(net, XP', YP');
end

training_time = toc;  % Record training time

% Evaluate network performance
fprintf('Evaluating network performance...\n');
[train_error, train_output] = nneval(net, XP(:, tr.trainInd)', YP(:, tr.trainInd)');
[val_error, val_output] = nneval(net, XP(:, tr.valInd)', YP(:, tr.valInd)');
[test_error, test_output] = nneval(net, XP(:, tr.testInd)', YP(:, tr.testInd)');

% Calculate R^2
train_r2 = calculateR2(YP(:, tr.trainInd)', train_output);
val_r2 = calculateR2(YP(:, tr.valInd)', val_output);
test_r2 = calculateR2(YP(:, tr.testInd)', test_output);

% Save results
results = struct();
results.net = net;
results.tr = tr;
results.training_time = training_time;
results.train_error = train_error;
results.val_error = val_error;
results.test_error = test_error;
results.train_r2 = train_r2;
results.val_r2 = val_r2;
results.test_r2 = test_r2;
results.config = current_config;

save(fullfile(result_dir, 'Results_trained.mat'), '-struct', 'results');

% Generate performance evaluation figures
generatePerformanceFigures(net, tr, train_output, val_output, test_output, ...
    YP(:, tr.trainInd)', YP(:, tr.valInd)', YP(:, tr.testInd)', figure_dir);

% Create performance summary
performance_summary = struct();
performance_summary.config_id = config_id;
performance_summary.train_mse = train_error;
performance_summary.val_mse = val_error;
performance_summary.test_mse = test_error;
performance_summary.train_r2 = train_r2;
performance_summary.val_r2 = val_r2;
performance_summary.test_r2 = test_r2;
performance_summary.epochs = length(tr.epoch);
performance_summary.training_time = training_time;
performance_summary.stop_reason = tr.stop;

% Save performance summary
perf_table = struct2table(performance_summary);
writetable(perf_table, fullfile(result_dir, 'performance_summary.csv'));

fprintf('Configuration %d training complete. Training MSE: %.6f, Validation MSE: %.6f, Test MSE: %.6f\n', ...
    config_id, train_error, val_error, test_error);
fprintf('Training R^2: %.4f, Validation R^2: %.4f, Test R^2: %.4f\n', ...
    train_r2, val_r2, test_r2);
fprintf('Training epochs: %d, Training time: %.2f seconds\n', length(tr.epoch), training_time);
end

function param_grid = generateParamGrid(config)
% Generate grid of all hyperparameter combinations
learning_rates = config.learning_rate;
momentums = config.momentum;
hidden_layers_configs = config.hidden_layers;
transfer_functions_configs = config.transfer_functions;
lr_increase_factors = config.lr_increase_factor;
lr_decrease_factors = config.lr_decrease_factor;
division_ratios = config.division_ratio;
max_fails = config.max_fail;
max_epochs_values = config.max_epochs;

% Initialize parameter grid
param_grid = struct([]);
idx = 1;

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
                                    param_grid(idx).learning_rate = learning_rates(lr_idx);
                                    param_grid(idx).momentum = momentums(mc_idx);
                                    param_grid(idx).hidden_layers = hidden_layers_configs{hl_idx};
                                    param_grid(idx).transfer_functions = transfer_functions_configs{tf_idx};
                                    param_grid(idx).lr_increase_factor = lr_increase_factors(lr_inc_idx);
                                    param_grid(idx).lr_decrease_factor = lr_decrease_factors(lr_dec_idx);
                                    param_grid(idx).division_ratio = division_ratios(div_idx, :);
                                    param_grid(idx).max_fail = max_fails(mf_idx);
                                    param_grid(idx).max_epochs = max_epochs_values(me_idx);
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
end

function r2 = calculateR2(actual, predicted)
% Calculate R^2 value
SST = sum((actual - mean(actual)).^2);
SSR = sum((predicted - mean(actual)).^2);
SSE = sum((actual - predicted).^2);
r2 = 1 - SSE/SST;
end

function generatePerformanceFigures(net, tr, train_output, val_output, test_output, ...
    train_targets, val_targets, test_targets, figure_dir)
% Generate performance evaluation figures

% Training performance plot
figure('visible', 'off');
subplot(2, 1, 1);
semilogy(tr.epoch, tr.perf, 'b-', 'LineWidth', 2);
hold on;
if ~isempty(tr.vperf)
    semilogy(tr.epoch, tr.vperf, 'g-', 'LineWidth', 2);
end
if ~isempty(tr.tperf)
    semilogy(tr.epoch, tr.tperf, 'r-', 'LineWidth', 2);
end
legend('Training', 'Validation', 'Testing');
title('Mean Squared Error (MSE) Performance');
xlabel('Epochs');
ylabel('MSE (log scale)');
grid on;

subplot(2, 1, 2);
plot(tr.epoch, tr.gradient, 'b-', 'LineWidth', 2);
title('Gradient');
xlabel('Epochs');
ylabel('Gradient');
grid on;

saveas(gcf, fullfile(figure_dir, 'Fig1_TrainingPerformance.fig'));
saveas(gcf, fullfile(figure_dir, 'Fig1_TrainingPerformance.png'));

% Regression plot - Training set
figure('visible', 'off');
plotregression(train_targets, train_output, 'Training');
saveas(gcf, fullfile(figure_dir, 'Fig2_TrainingRegression.fig'));
saveas(gcf, fullfile(figure_dir, 'Fig2_TrainingRegression.png'));

% Regression plot - Validation set
figure('visible', 'off');
plotregression(val_targets, val_output, 'Validation');
saveas(gcf, fullfile(figure_dir, 'Fig3_ValidationRegression.fig'));
saveas(gcf, fullfile(figure_dir, 'Fig3_ValidationRegression.png'));

% Regression plot - Test set
figure('visible', 'off');
plotregression(test_targets, test_output, 'Testing');
saveas(gcf, fullfile(figure_dir, 'Fig4_TestRegression.fig'));
saveas(gcf, fullfile(figure_dir, 'Fig4_TestRegression.png'));

% Combined regression plot
figure('visible', 'off');
plotregression(train_targets, train_output, 'Training', ...
               val_targets, val_output, 'Validation', ...
               test_targets, test_output, 'Testing', ...
               [train_targets; val_targets; test_targets], ...
               [train_output; val_output; test_output], 'All');
saveas(gcf, fullfile(figure_dir, 'Fig5_AllRegression.fig'));
saveas(gcf, fullfile(figure_dir, 'Fig5_AllRegression.png'));

% Error histogram
figure('visible', 'off');
train_errors = train_targets - train_output;
val_errors = val_targets - val_output;
test_errors = test_targets - test_output;

subplot(3, 1, 1);
histogram(train_errors, 50);
title('Training Set Error Distribution');
xlabel('Error');
ylabel('Frequency');
grid on;

subplot(3, 1, 2);
histogram(val_errors, 50);
title('Validation Set Error Distribution');
xlabel('Error');
ylabel('Frequency');
grid on;

subplot(3, 1, 3);
histogram(test_errors, 50);
title('Test Set Error Distribution');
xlabel('Error');
ylabel('Frequency');
grid on;

saveas(gcf, fullfile(figure_dir, 'Fig6_ErrorHistogram.fig'));
saveas(gcf, fullfile(figure_dir, 'Fig6_ErrorHistogram.png'));
end
