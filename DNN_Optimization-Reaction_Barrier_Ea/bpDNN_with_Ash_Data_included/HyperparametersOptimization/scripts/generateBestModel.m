function generateBestModel(best_config_id)
% GENERATEBESTMODEL Retrain and save the final model based on the best hyperparameter configuration
% Input parameters:
%   best_config_id: ID of the best configuration from hyperparameter optimization

% Define paths
current_dir = pwd;
results_dir = fullfile(current_dir, 'results');
best_model_dir = fullfile(current_dir, 'best_model');
data_dir = fullfile(current_dir, 'data');

% Create output directory if it doesn't exist
if ~exist(best_model_dir, 'dir')
    mkdir(best_model_dir);
end

figures_dir = fullfile(best_model_dir, 'Figures');
if ~exist(figures_dir, 'dir')
    mkdir(figures_dir);
end

% Find best configuration directory
best_config_dir = fullfile(results_dir, sprintf('config_%d', best_config_id));
if ~exist(best_config_dir, 'dir')
    error('Best configuration directory (ID: %d) not found. Please provide a valid configuration ID.', best_config_id);
end

% Load prepared data
data_file = fullfile(data_dir, 'prepared_data.mat');
if ~exist(data_file, 'file')
    error('Prepared data file not found. Run prepare_data.m first.');
end

% Load data
load(data_file, 'Variables', 'Ea');

% Load best configuration
config_info_file = fullfile(best_config_dir, 'config_info.csv');
if ~exist(config_info_file, 'file')
    error('Configuration info file not found in best configuration directory.');
end

config_info = readtable(config_info_file);

% Print best configuration information
fprintf('Retraining final model using best configuration (ID: %d)\n', best_config_id);
fprintf('Learning Rate: %.4f\n', config_info.learning_rate);
fprintf('Momentum: %.4f\n', config_info.momentum);
fprintf('Learning Rate Increase Factor: %.4f\n', config_info.lr_increase_factor);
fprintf('Learning Rate Decrease Factor: %.4f\n', config_info.lr_decrease_factor);
fprintf('Maximum Validation Failures: %d\n', config_info.max_fail);
fprintf('Maximum Training Epochs: %d\n', config_info.max_epochs);

% Parse hidden layers structure
hidden_layers_str = config_info.hidden_layers{1};
hidden_layers_str = strrep(hidden_layers_str, '[', '');
hidden_layers_str = strrep(hidden_layers_str, ']', '');
hidden_layers_parts = strsplit(hidden_layers_str, ',');
hidden_layers = zeros(1, length(hidden_layers_parts));
for i = 1:length(hidden_layers_parts)
    hidden_layers(i) = str2double(hidden_layers_parts{i});
end

% Parse transfer functions
transfer_functions_str = config_info.transfer_functions{1};
transfer_functions_str = strrep(transfer_functions_str, '[', '');
transfer_functions_str = strrep(transfer_functions_str, ']', '');
transfer_functions = strsplit(transfer_functions_str, ',');
for i = 1:length(transfer_functions)
    transfer_functions{i} = strtrim(transfer_functions{i});
    transfer_functions{i} = strrep(transfer_functions{i}, '"', '');
    transfer_functions{i} = strrep(transfer_functions{i}, '''', '');
end

% Parse data division ratio
division_ratio_str = config_info.division_ratio{1};
division_ratio_str = strrep(division_ratio_str, '[', '');
division_ratio_str = strrep(division_ratio_str, ']', '');
division_ratio_parts = strsplit(division_ratio_str, ',');
division_ratio = zeros(1, length(division_ratio_parts));
for i = 1:length(division_ratio_parts)
    division_ratio(i) = str2double(division_ratio_parts{i});
end

% Create and configure network
net = nncreate(hidden_layers);

% Set training parameters
net.trainParam.mc = config_info.momentum;  % Momentum coefficient
net.trainParam.lr = config_info.learning_rate;  % Learning rate
net.trainParam.lr_inc = config_info.lr_increase_factor;  % Learning rate increase factor
net.trainParam.lr_dec = config_info.lr_decrease_factor;  % Learning rate decrease factor
net.trainParam.max_fail = config_info.max_fail;  % Maximum validation failures
net.trainParam.epoch = config_info.max_epochs;  % Maximum training epochs

% Set transfer functions
for i = 1:length(hidden_layers)
    net.layer{i}.transferFcn = transfer_functions{i};
end
net.layer{end}.transferFcn = transfer_functions{end};

% Set data division ratio
net.divideFcn = 'dividerand';
net.divideParam.trainRatio = division_ratio(1);
net.divideParam.valRatio = division_ratio(2);
net.divideParam.testRatio = division_ratio(3);

% Set normalization
net.processFcn = 'mapminmax';
net.processParam.ymax = 1;
net.processParam.ymin = -1;

% Prepare input and target data
XP = Variables;  % Input data
YP = Ea;  % Target data

% Train network
fprintf('Starting final model training...\n');
tic;  % Record start time
[net, tr] = nntrain(net, XP', YP');
training_time = toc;  % Record training time

% Evaluate network performance
fprintf('Evaluating final model performance...\n');
[train_error, train_output] = nneval(net, XP(:, tr.trainInd)', YP(:, tr.trainInd)');
[val_error, val_output] = nneval(net, XP(:, tr.valInd)', YP(:, tr.valInd)');
[test_error, test_output] = nneval(net, XP(:, tr.testInd)', YP(:, tr.testInd)');

% Calculate R^2
train_r2 = calculateR2(YP(:, tr.trainInd)', train_output);
val_r2 = calculateR2(YP(:, tr.valInd)', val_output);
test_r2 = calculateR2(YP(:, tr.testInd)', test_output);

% Generate performance evaluation figures
generatePerformanceFigures(net, tr, train_output, val_output, test_output, ...
    YP(:, tr.trainInd)', YP(:, tr.valInd)', YP(:, tr.testInd)', figures_dir);

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
results.config = struct(...
    'learning_rate', config_info.learning_rate, ...
    'momentum', config_info.momentum, ...
    'hidden_layers', hidden_layers, ...
    'transfer_functions', {transfer_functions}, ...
    'lr_increase_factor', config_info.lr_increase_factor, ...
    'lr_decrease_factor', config_info.lr_decrease_factor, ...
    'division_ratio', division_ratio, ...
    'max_fail', config_info.max_fail, ...
    'max_epochs', config_info.max_epochs);

save(fullfile(best_model_dir, 'Results_trained.mat'), '-struct', 'results');

% Copy necessary MATLAB functions
src_dir = fullfile(current_dir, 'matlab_src');
files_to_copy = {'nnpredict.m', 'nnpreprocess.m', 'nnpostprocess.m', 'nnff.m', 'nnfindbest.m'};
for i = 1:length(files_to_copy)
    if exist(fullfile(src_dir, files_to_copy{i}), 'file')
        copyfile(fullfile(src_dir, files_to_copy{i}), fullfile(best_model_dir, files_to_copy{i}));
        fprintf('Copied %s\n', files_to_copy{i});
    else
        fprintf('Warning: %s not found in matlab_src directory\n', files_to_copy{i});
    end
end

% Copy sample input data if available
sample_files = {'SampleIn_pred.mat', 'SampleIn_scenario1.mat', 'SampleIn_scenario2.mat', 'SampleIn_scenario3.mat'};
for i = 1:length(sample_files)
    if exist(fullfile(src_dir, sample_files{i}), 'file')
        copyfile(fullfile(src_dir, sample_files{i}), fullfile(best_model_dir, sample_files{i}));
        fprintf('Copied %s\n', sample_files{i});
    end
end

% Copy script for prediction demonstration if available
if exist(fullfile(src_dir, 'script4PDP.m'), 'file')
    copyfile(fullfile(src_dir, 'script4PDP.m'), fullfile(best_model_dir, 'script4PDP.m'));
    fprintf('Copied script4PDP.m\n');
end

% Create README file
fid = fopen(fullfile(best_model_dir, 'README.txt'), 'w');
fprintf(fid, 'Optimized BP-DNN Model\n');
fprintf(fid, '==============================\n\n');
fprintf(fid, 'This model was selected as the best model through an extensive hyperparameter optimization process.\n\n');
fprintf(fid, 'Model Performance:\n');
fprintf(fid, '  Training MSE: %.6f\n', train_error);
fprintf(fid, '  Validation MSE: %.6f\n', val_error);
fprintf(fid, '  Testing MSE: %.6f\n', test_error);
fprintf(fid, '  Training R^2: %.4f\n', train_r2);
fprintf(fid, '  Validation R^2: %.4f\n', val_r2);
fprintf(fid, '  Testing R^2: %.4f\n', test_r2);
fprintf(fid, '  Training Epochs: %d\n', length(tr.epoch));
fprintf(fid, '  Training Time: %.2f seconds\n', training_time);
fprintf(fid, '\n');
fprintf(fid, 'Hyperparameters:\n');
fprintf(fid, '  Learning Rate: %.4f\n', config_info.learning_rate);
fprintf(fid, '  Momentum Coefficient: %.4f\n', config_info.momentum);
fprintf(fid, '  Hidden Layer Structure: %s\n', config_info.hidden_layers{1});
fprintf(fid, '  Transfer Functions: %s\n', config_info.transfer_functions{1});
fprintf(fid, '  Learning Rate Increase Factor: %.4f\n', config_info.lr_increase_factor);
fprintf(fid, '  Learning Rate Decrease Factor: %.4f\n', config_info.lr_decrease_factor);
fprintf(fid, '  Data Division Ratio: %s\n', config_info.division_ratio{1});
fprintf(fid, '  Maximum Validation Failures: %d\n', config_info.max_fail);
fprintf(fid, '  Maximum Training Epochs: %d\n', config_info.max_epochs);
fprintf(fid, '\n');
fprintf(fid, 'Usage Instructions:\n');
fprintf(fid, '1. Load Results_trained.mat to get the trained model\n');
fprintf(fid, '2. Use the nnpredict function for predictions\n');
fprintf(fid, '3. See script4PDP.m for example prediction code\n');
fclose(fid);

fprintf('Final model training and saving completed.\n');
fprintf('Model Performance:\n');
fprintf('  Training MSE: %.6f\n', train_error);
fprintf('  Validation MSE: %.6f\n', val_error);
fprintf('  Testing MSE: %.6f\n', test_error);
fprintf('  Training R^2: %.4f\n', train_r2);
fprintf('  Validation R^2: %.4f\n', val_r2);
fprintf('  Testing R^2: %.4f\n', test_r2);
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
print(gcf, fullfile(figure_dir, 'Fig1_TrainingPerformance.png'), '-dpng', '-r600');
print(gcf, fullfile(figure_dir, 'Fig1_TrainingPerformance.eps'), '-depsc2', '-r600');

% Regression plot - Training set
figure('visible', 'off');
plotregression(train_targets, train_output, 'Training');
saveas(gcf, fullfile(figure_dir, 'Fig2_TrainingRegression.fig'));
print(gcf, fullfile(figure_dir, 'Fig2_TrainingRegression.png'), '-dpng', '-r600');
print(gcf, fullfile(figure_dir, 'Fig2_TrainingRegression.eps'), '-depsc2', '-r600');

% Regression plot - Validation set
figure('visible', 'off');
plotregression(val_targets, val_output, 'Validation');
saveas(gcf, fullfile(figure_dir, 'Fig3_ValidationRegression.fig'));
print(gcf, fullfile(figure_dir, 'Fig3_ValidationRegression.png'), '-dpng', '-r600');
print(gcf, fullfile(figure_dir, 'Fig3_ValidationRegression.eps'), '-depsc2', '-r600');

% Regression plot - Test set
figure('visible', 'off');
plotregression(test_targets, test_output, 'Testing');
saveas(gcf, fullfile(figure_dir, 'Fig4_TestRegression.fig'));
print(gcf, fullfile(figure_dir, 'Fig4_TestRegression.png'), '-dpng', '-r600');
print(gcf, fullfile(figure_dir, 'Fig4_TestRegression.eps'), '-depsc2', '-r600');

% Combined regression plot
figure('visible', 'off');
plotregression(train_targets, train_output, 'Training', ...
               val_targets, val_output, 'Validation', ...
               test_targets, test_output, 'Testing', ...
               [train_targets; val_targets; test_targets], ...
               [train_output; val_output; test_output], 'All');
saveas(gcf, fullfile(figure_dir, 'Fig5_AllRegression.fig'));
print(gcf, fullfile(figure_dir, 'Fig5_AllRegression.png'), '-dpng', '-r600');
print(gcf, fullfile(figure_dir, 'Fig5_AllRegression.eps'), '-depsc2', '-r600');

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
print(gcf, fullfile(figure_dir, 'Fig6_ErrorHistogram.png'), '-dpng', '-r600');
print(gcf, fullfile(figure_dir, 'Fig6_ErrorHistogram.eps'), '-depsc2', '-r600');

% Copy figures to tif format for publication quality
for i = 1:6
    fig_file = fullfile(figure_dir, sprintf('Fig%d_*.fig', i));
    fig_files = dir(fig_file);
    
    for j = 1:length(fig_files)
        [~, fig_name, ~] = fileparts(fig_files(j).name);
        tif_name = [fig_name, '.tif'];
          % Create high-resolution TIF file
        fig = openfig(fullfile(figure_dir, fig_files(j).name), 'invisible');
        print(fig, fullfile(figure_dir, tif_name), '-dtiff', '-r600');
        close(fig);
    end
end
end
