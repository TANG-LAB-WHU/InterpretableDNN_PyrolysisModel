%% Model Parameter Optimization for bpDNN4PyroProd
% This script performs hyperparameter optimization to find the best parameters
% for the neural network model used in pyrolysis product yield prediction.
% It uses grid search with cross-validation to systematically evaluate
% different parameter combinations.

clc;
fprintf('Starting neural network parameter optimization...\n');

%% Load and prepare data (same as in bpDNN4PyroProd.m)
fprintf('Loading and preparing data...\n');

% Load feedstock encoding data
load('CopyrolysisFeedstock.mat');

FeedType_size = size(CopyrolysisFeedstockTag, 1);
Total_FeedID_size = size(Total_FeedID, 1);
Total_MixingRatio_size = size(Total_MixingRatio, 1);

MixingFeedID = zeros(Total_FeedID_size, FeedType_size);
MixingRatio = zeros(Total_MixingRatio_size, FeedType_size);

for i = 1 : Total_MixingRatio_size
    MixingFeedID(i, FeedstockIndex(i, 1)) = Total_FeedID(i, 1);
    MixingFeedID(i, FeedstockIndex(i, 2)) = Total_FeedID(i, 2);
    MixingRatio(i, FeedstockIndex(i, 1)) = Total_MixingRatio(i, 1);
    MixingRatio(i, FeedstockIndex(i, 2)) = Total_MixingRatio(i, 2);
end

Feedstock4training = [MixingFeedID MixingRatio];

% Generate input and target dataset
PreparedInputData = readmatrix('RawInputData.xlsx');

% Input data
Location =  PreparedInputData(:, 1); 
VolatileMatters =  PreparedInputData(:, 2); FixedCarbon =  PreparedInputData(:, 3); Ash =  PreparedInputData(:, 4);
C =  PreparedInputData(:, 5); H =  PreparedInputData(:, 6); N =  PreparedInputData(:, 7); O =  PreparedInputData(:, 8); S =  PreparedInputData(:, 9);
TargetTemperature =  PreparedInputData(:, 10); ReactionTime =  PreparedInputData(:, 11); HeatingRate =  PreparedInputData(:, 12);
ReactorType =  PreparedInputData(:, 13);
Variables0 = [Location VolatileMatters FixedCarbon Ash C H N O S TargetTemperature ReactionTime...
    HeatingRate ReactorType];
Variables = [Variables0 Feedstock4training];

% Target data
CharYield = PreparedInputData(:, 14);
LiquidYield = PreparedInputData(:, 15);
GasYield = PreparedInputData(:, 16);
ProductsYield = [CharYield LiquidYield GasYield];

% Name Variables as RESPONSE and PREDICTOR
Target = table(ProductsYield(:, 1), ProductsYield(:, 2), ProductsYield(:, 3));
Target.Properties.VariableNames([1 2 3]) = {'Char/%' 'Liquid/%' 'Gas/%'};

% Assign training dataset
input = Variables'; 
target = Target.Variables';

%% Define parameter search space
fprintf('Setting up parameter grid search...\n');

% Create directory for optimization results
if ~exist('output/optimization_results', 'dir')
    mkdir('output/optimization_results');
end

% Define parameter grid
% Format: {parameter_name, [possible_values]}
param_grid = {
    {'hidden_layers', {
        [20, 20, 20],             % 3 hidden layers with 20 neurons each
        [30, 30, 30],             % 3 hidden layers with 30 neurons each
        [37, 37, 37],             % 3 hidden layers with 37 neurons each
        [37, 37, 37, 37],         % 4 hidden layers with 37 neurons each
        [37, 37, 37, 37, 37],     % 5 hidden layers with 37 neurons each (original)
        [40, 40, 40, 40, 40],     % 5 hidden layers with 40 neurons each
        [50, 40, 30, 40, 50],     % 5 hidden layers with different neuron counts
        [50, 50, 50]              % 3 hidden layers with 50 neurons each
    }},
    {'learning_rate', [0.05, 0.1, 0.25, 0.5, 0.75, 0.85, 0.95]},
    {'momentum', [0.7, 0.8, 0.9, 0.95, 0.97, 0.99]},
    {'transfer_fcn', {'tansig', 'logsig', 'elliotsig'}}
};

% Setup cross-validation
num_folds = 5;
data_size = size(input, 2);
fold_size = floor(data_size / num_folds);

% Prepare results storage
results = struct('params', {}, 'performance', {}, 'model', {});
best_performance = Inf;
best_params = [];
best_model = [];

% Prepare results table
results_table = table('Size', [0, 7], 'VariableTypes', {'cell', 'double', 'double', 'string', 'double', 'double', 'double'}, ...
                     'VariableNames', {'HiddenLayers', 'LearningRate', 'Momentum', 'TransferFunction', 'TrainMSE', 'ValMSE', 'TestMSE'});

%% Generate parameter combinations
param_combinations = generate_param_combinations(param_grid, 1, {});
num_combinations = length(param_combinations);

fprintf('Starting grid search with %d parameter combinations and %d-fold cross-validation...\n', num_combinations, num_folds);

%% Perform grid search with cross-validation
result_idx = 1;
for combo_idx = 1:num_combinations
    current_params = param_combinations{combo_idx};
    
    % Extract parameters
    hidden_layers = current_params{1}{2};
    learning_rate = current_params{2}{2};
    momentum = current_params{3}{2};
    transfer_fcn = current_params{4}{2};
    
    fprintf('\nEvaluating parameter combination %d/%d:\n', combo_idx, num_combinations);
    fprintf('Hidden Layers: '); disp(hidden_layers);
    fprintf('Learning Rate: %.3f, Momentum: %.3f, Transfer Function: %s\n', learning_rate, momentum, transfer_fcn);
    
    % Initialize performance metrics for cross-validation
    cv_train_perf = zeros(num_folds, 1);
    cv_val_perf = zeros(num_folds, 1);
    cv_test_perf = zeros(num_folds, 1);
    
    % Cross-validation loop
    for fold = 1:num_folds
        fprintf('  Fold %d/%d... ', fold, num_folds);
        
        % Create fold indices
        test_start = (fold-1) * fold_size + 1;
        test_end = min(fold * fold_size, data_size);
        test_indices = test_start:test_end;
        
        % For validation, use the next fold
        val_start = mod(test_end, data_size) + 1;
        val_end = min(val_start + fold_size - 1, data_size);
        val_indices = val_start:val_end;
        
        % Everything else is for training
        train_indices = setdiff(1:data_size, [test_indices, val_indices]);
        
        % Split data
        train_input = input(:, train_indices);
        train_target = target(:, train_indices);
        val_input = input(:, val_indices);
        val_target = target(:, val_indices);
        test_input = input(:, test_indices);
        test_target = target(:, test_indices);
        
        % Create and configure network
        net = nncreate(hidden_layers);
        
        % Set training parameters
        net.trainParam.mc = momentum;
        net.trainParam.lr = learning_rate;
        net.trainParam.lr_inc = 0.15;
        net.trainParam.lr_dec = 1.40;
        net.trainParam.goal = 1e-6;
        net.trainParam.min_grad = 1e-5;
        net.trainParam.max_fail = 10;
        net.trainParam.epoch = 1000; % Use fewer epochs for optimization to save time
        
        % Set transfer functions
        Nl = size(hidden_layers, 2) + 1;
        for i = 1:Nl
            if i == Nl
                net.layer{i}.transferFcn = 'purelin'; % Output layer
            else
                net.layer{i}.transferFcn = transfer_fcn; % Hidden layers
            end
        end
        
        % Set other network parameters
        net.processFcn = 'mapminmax';
        net.processParam.ymax = 1;
        net.processParam.ymin = -1;
        
        % Disable automatic division, we're doing manual CV
        net.divideFcn = 'dividetrain';
        
        % Set performance function and disable real-time display
        net.performFcn = 'mse';
        net.adaptFcn = 'none';
        net.trainParam.showCommandLine = false;
        net.trainParam.showWindow = false;
        
        % Train the network
        [trained_net, tr] = nntrain(net, train_input, train_target);
        
        % Evaluate performance
        train_out = nnpredict(trained_net, train_input);
        train_perf = nneval(trained_net, train_input, train_target);
        cv_train_perf(fold) = train_perf;
        
        val_out = nnpredict(trained_net, val_input);
        val_perf = nneval(trained_net, val_input, val_target);
        cv_val_perf(fold) = val_perf;
        
        test_out = nnpredict(trained_net, test_input);
        test_perf = nneval(trained_net, test_input, test_target);
        cv_test_perf(fold) = test_perf;
        
        fprintf('Train MSE: %.6f, Val MSE: %.6f, Test MSE: %.6f\n', train_perf, val_perf, test_perf);
    end
    
    % Calculate average performance across folds
    avg_train_perf = mean(cv_train_perf);
    avg_val_perf = mean(cv_val_perf);
    avg_test_perf = mean(cv_test_perf);
    
    fprintf('Average performance - Train MSE: %.6f, Val MSE: %.6f, Test MSE: %.6f\n', ...
            avg_train_perf, avg_val_perf, avg_test_perf);
    
    % Add to results table
    hidden_layers_str = mat2str(hidden_layers);
    results_table(result_idx, :) = {hidden_layers, learning_rate, momentum, transfer_fcn, ...
                                    avg_train_perf, avg_val_perf, avg_test_perf};
    result_idx = result_idx + 1;
    
    % Check if this is the best model based on validation performance
    if avg_val_perf < best_performance
        best_performance = avg_val_perf;
        best_params = current_params;
        
        % Train a full model with the best parameters
        fprintf('New best model found! Training full model with these parameters...\n');
        
        % Create and configure best network
        best_net = nncreate(hidden_layers);
        
        % Set training parameters
        best_net.trainParam.mc = momentum;
        best_net.trainParam.lr = learning_rate;
        best_net.trainParam.lr_inc = 0.15;
        best_net.trainParam.lr_dec = 1.40;
        best_net.trainParam.goal = 1e-6;
        best_net.trainParam.min_grad = 1e-5;
        best_net.trainParam.max_fail = 20;
        best_net.trainParam.epoch = 2000; % More epochs for final model
        
        % Set transfer functions
        for i = 1:Nl
            if i == Nl
                best_net.layer{i}.transferFcn = 'purelin'; % Output layer
            else
                best_net.layer{i}.transferFcn = transfer_fcn; % Hidden layers
            end
        end
        
        % Set other network parameters
        best_net.processFcn = 'mapminmax';
        best_net.processParam.ymax = 1;
        best_net.processParam.ymin = -1;
        
        % Use standard division for final model
        best_net.divideFcn = 'dividerand';
        best_net.divideParam.trainRatio = 0.80;
        best_net.divideParam.valRatio = 0.10;
        best_net.divideParam.testRatio = 0.10;
        
        % Set performance function
        best_net.performFcn = 'mse';
        best_net.adaptFcn = 'none';
        best_net.trainParam.showCommandLine = true;
        best_net.trainParam.showWindow = true;
        
        % Train the best model
        [best_model, best_tr] = nntrain(best_net, input, target);
        
        % Save the best model
        best_params_struct = struct();
        best_params_struct.hidden_layers = hidden_layers;
        best_params_struct.learning_rate = learning_rate;
        best_params_struct.momentum = momentum;
        best_params_struct.transfer_fcn = transfer_fcn;
        best_params_struct.train_mse = avg_train_perf;
        best_params_struct.val_mse = avg_val_perf;
        best_params_struct.test_mse = avg_test_perf;
        
        save('output/optimization_results/best_model.mat', 'best_model', 'best_params_struct', 'best_tr');
    end
end

%% Save optimization results
writetable(results_table, 'output/optimization_results/parameter_search_results.csv');

% Sort results by validation MSE
sorted_results = sortrows(results_table, 'ValMSE', 'ascend');
writetable(sorted_results, 'output/optimization_results/sorted_parameter_search_results.csv');

% Save top 5 models
top5_results = sorted_results(1:min(5, height(sorted_results)), :);
writetable(top5_results, 'output/optimization_results/top5_models.csv');

% Create optimization summary figure
figure('Name', 'Parameter Optimization Results');

% Learning rate vs MSE for different layer configurations
subplot(2, 2, 1);
gscatter(results_table.LearningRate, results_table.ValMSE, cellfun(@mat2str, results_table.HiddenLayers, 'UniformOutput', false));
xlabel('Learning Rate');
ylabel('Validation MSE');
title('Effect of Learning Rate on Performance');
legend('Location', 'best');

% Momentum vs MSE
subplot(2, 2, 2);
gscatter(results_table.Momentum, results_table.ValMSE, cellfun(@mat2str, results_table.HiddenLayers, 'UniformOutput', false));
xlabel('Momentum');
ylabel('Validation MSE');
title('Effect of Momentum on Performance');

% Transfer function comparison
subplot(2, 2, 3);
boxplot(results_table.ValMSE, results_table.TransferFunction);
xlabel('Transfer Function');
ylabel('Validation MSE');
title('Effect of Transfer Function on Performance');

% Training vs Validation performance
subplot(2, 2, 4);
scatter(results_table.TrainMSE, results_table.ValMSE, 50, 'filled');
xlabel('Training MSE');
ylabel('Validation MSE');
title('Training vs Validation Performance');
grid on;

% Save the figure
savefig('output/optimization_results/optimization_summary.fig');
saveas(gcf, 'output/optimization_results/optimization_summary.png');

%% Create report with best parameters
fprintf('\n\n======= Optimization Results Summary =======\n');
fprintf('Best parameters found:\n');
fprintf('  Hidden Layers: '); disp(best_params{1}{2});
fprintf('  Learning Rate: %.3f\n', best_params{2}{2});
fprintf('  Momentum: %.3f\n', best_params{3}{2});
fprintf('  Transfer Function: %s\n', best_params{4}{2});
fprintf('  Validation MSE: %.6f\n', best_performance);

% Create a text report
fid = fopen('output/optimization_results/optimization_report.txt', 'w');
fprintf(fid, 'Neural Network Parameter Optimization Report\n');
fprintf(fid, '=======================================\n\n');
fprintf(fid, 'Optimization completed on: %s\n\n', datestr(now));
fprintf(fid, 'Best parameters found:\n');
fprintf(fid, '  Hidden Layers: %s\n', mat2str(best_params{1}{2}));
fprintf(fid, '  Learning Rate: %.3f\n', best_params{2}{2});
fprintf(fid, '  Momentum: %.3f\n', best_params{3}{2});
fprintf(fid, '  Transfer Function: %s\n', best_params{4}{2});
fprintf(fid, '  Training MSE: %.6f\n', avg_train_perf);
fprintf(fid, '  Validation MSE: %.6f\n', best_performance);
fprintf(fid, '  Testing MSE: %.6f\n\n', avg_test_perf);
fprintf(fid, 'Top 5 Parameter Combinations:\n');
fprintf(fid, '---------------------------\n');
for i = 1:min(5, height(sorted_results))
    fprintf(fid, 'Rank %d:\n', i);
    fprintf(fid, '  Hidden Layers: %s\n', mat2str(sorted_results.HiddenLayers{i}));
    fprintf(fid, '  Learning Rate: %.3f\n', sorted_results.LearningRate(i));
    fprintf(fid, '  Momentum: %.3f\n', sorted_results.Momentum(i));
    fprintf(fid, '  Transfer Function: %s\n', sorted_results.TransferFunction{i});
    fprintf(fid, '  Validation MSE: %.6f\n\n', sorted_results.ValMSE(i));
end
fclose(fid);

fprintf('\nOptimization completed. Results saved to output/optimization_results/\n');
fprintf('The best model has been saved as best_model.mat\n');
fprintf('To use this model for training, run train_with_optimal_params.m\n');

%% Helper function to generate parameter combinations
function combinations = generate_param_combinations(param_grid, current_idx, current_combo)
    if current_idx > length(param_grid)
        combinations = {current_combo};
        return;
    end
    
    param_name = param_grid{current_idx}{1};
    param_values = param_grid{current_idx}{2};
    
    combinations = {};
    for i = 1:length(param_values)
        value = param_values{i};
        if ~iscell(value)
            value = {value};
        end
        new_combo = [current_combo, {{param_name, value{1}}}];
        sub_combinations = generate_param_combinations(param_grid, current_idx + 1, new_combo);
        combinations = [combinations, sub_combinations];
    end
end 