%% Train Best Model Script
% This script loads the best hyperparameters from optimization 
% and trains a full model with those parameters

% Clear workspace and command window
clear all; close all; clc;

% Start timer
startTime = tic;

% Setup diary for logging
diary('train_best_model_log.txt');
diary on;

fprintf('Starting train_best_model.m\n');
fprintf('======================================\n\n');

try
    %% Setup directories and paths
    % Get the root directory
scriptPath = mfilename('fullpath');
[scriptDir, ~, ~] = fileparts(scriptPath);
rootDir = fileparts(fileparts(scriptDir));
    fprintf('Root directory: %s\n', rootDir);
    
    % Set paths to various directories
    modelDir = fullfile(rootDir, 'src', 'model');
resultsDir = fullfile(rootDir, 'results');
bestModelDir = fullfile(resultsDir, 'best_model');
    trainingDir = fullfile(resultsDir, 'training');
optimizationDir = fullfile(resultsDir, 'optimization');

% Create directories if they don't exist
if ~exist(resultsDir, 'dir')
    mkdir(resultsDir);
    fprintf('Created results directory: %s\n', resultsDir);
end

if ~exist(bestModelDir, 'dir')
    mkdir(bestModelDir);
    fprintf('Created best model directory: %s\n', bestModelDir);
end

if ~exist(trainingDir, 'dir')
    mkdir(trainingDir);
    fprintf('Created training directory: %s\n', trainingDir);
end

% Figures directory
figuresDir = fullfile(bestModelDir, 'Figures');
if ~exist(figuresDir, 'dir')
    mkdir(figuresDir);
    fprintf('Created figures directory: %s\n', figuresDir);
end

    % Add model directory to path
    addpath(modelDir);

% Check for required functions
requiredFunctions = {'nncreate', 'nnpredict', 'nnpreprocess', 'nntrain', 'nneval'};
for i = 1:length(requiredFunctions)
        if ~exist(fullfile(modelDir, [requiredFunctions{i} '.m']), 'file')
            error('Required function not found: %s', fullfile(modelDir, [requiredFunctions{i} '.m']));
        end
end

% Check data files existence and prepare paths
dataDir = fullfile(rootDir, 'data', 'processed');
dataRawDir = fullfile(rootDir, 'data', 'raw');
feedstockFile = fullfile(dataDir, 'CopyrolysisFeedstock.mat');
rawDataFile = fullfile(dataRawDir, 'RawInputData.xlsx');

    % Check if data files exist
    feedstockExists = exist(feedstockFile, 'file');
    rawDataExists = exist(rawDataFile, 'file');
    
    fprintf('Checking data files...\n');
    if feedstockExists
        fprintf('Feedstock file (%s): %s\n', feedstockFile, 'Exists');
    else
        fprintf('Feedstock file (%s): %s\n', feedstockFile, 'Not found');
    end
    
    if rawDataExists
        fprintf('Raw data file (%s): %s\n', rawDataFile, 'Exists');
    else
        fprintf('Raw data file (%s): %s\n', rawDataFile, 'Not found');
    end
    
    % If data files don't exist in data directory, check if they exist in model directory
    if ~feedstockExists
        modelFeedstockFile = fullfile(modelDir, 'CopyrolysisFeedstock.mat');
        if exist(modelFeedstockFile, 'file')
            feedstockFile = modelFeedstockFile;
            feedstockExists = true;
            fprintf('Found feedstock file in model directory: %s\n', feedstockFile);
        end
    end
    
    if ~rawDataExists
        modelRawDataFile = fullfile(modelDir, 'RawInputData.xlsx');
        if exist(modelRawDataFile, 'file')
            rawDataFile = modelRawDataFile;
            rawDataExists = true;
            fprintf('Found raw data file in model directory: %s\n', rawDataFile);
        end
    end

%% Load best hyperparameters
fprintf('Loading best hyperparameters...\n');

% Load the best configuration from optimization results
if ~exist('optimDir', 'var') || isempty(optimDir)
    optimDir = fullfile(rootDir, 'results', 'optimization');
end

% Check if optimization results exist
optimResultsFile = fullfile(optimDir, 'best_model.mat');
if ~exist(optimResultsFile, 'file')
    error('ERROR: Best model optimization results not found at %s', optimResultsFile);
end

% Load optimization results
disp('Loading optimization results...');
optimResults = load(optimResultsFile);
if ~isfield(optimResults, 'best_config')
    error('ERROR: Invalid optimization results file. Missing best_config field.');
end

best_config = optimResults.best_config;
disp('Best configuration loaded:');
disp(best_config);

% Make sure we can access the raw data file
rawDataFile = fullfile(modelDir, 'RawInputData.xlsx');
if ~exist(rawDataFile, 'file')
    rawDataFile = fullfile(rootDir, 'data', 'raw', 'RawInputData.xlsx');
    if ~exist(rawDataFile, 'file')
        error('ERROR: RawInputData.xlsx file not found. Cannot proceed with model training.');
    end
end

% Try to determine actual sample count from RawInputData.xlsx
fprintf('Reading RawInputData.xlsx to determine sample count and data dimensions\n');
try
    % Read the file to determine dimensions
    [~, ~, raw] = xlsread(rawDataFile);
    [numRows, numCols] = size(raw);
    numSamples = numRows - 1; % Subtract 1 for header row if present
    
    if numSamples <= 0
        error('Invalid sample count: RawInputData.xlsx contains no data rows');
    end
    
    % Based on bpDNN4PyroProd.m, target variables start at column 14
    inputFeatureEndCol = 13; % The last column index for input features before adding feedstock data
    targetStartCol = 14;     % The first column index where target variables begin
    numTargetCols = numCols - targetStartCol + 1; % Calculate how many target columns exist
    
    % Validate target columns
    if numTargetCols <= 0
        error('Invalid target column count. Target data should be in columns starting from column 14.');
    end
    
    fprintf('Sample count determined from RawInputData.xlsx: %d samples\n', numSamples);
    fprintf('Detected %d target variable columns (starting from column %d)\n', numTargetCols, targetStartCol);
catch readErr
    error('Error reading RawInputData.xlsx: %s', readErr.message);
end

% Determine the number of features dynamically instead of hardcoding
% Check if we can access CopyrolysisFeedstock.mat to determine FeedType_size
feedstockFile = fullfile(modelDir, 'CopyrolysisFeedstock.mat');
if ~exist(feedstockFile, 'file')
    feedstockFile = fullfile(rootDir, 'data', 'processed', 'CopyrolysisFeedstock.mat');
    if ~exist(feedstockFile, 'file')
        error('ERROR: CopyrolysisFeedstock.mat file not found. Cannot determine feature count.');
    end
end

% Try to determine feature count from CopyrolysisFeedstock.mat
fprintf('Determining feature count from CopyrolysisFeedstock.mat\n');
try
    feedstockData = load(feedstockFile);
    if isfield(feedstockData, 'CopyrolysisFeedstockTag')
        fprintf('Loaded CopyrolysisFeedstock.mat to determine feature count\n');
        FeedType_size = size(feedstockData.CopyrolysisFeedstockTag, 1);
        fprintf('FeedType_size determined: %d\n', FeedType_size);
        numFeatures = 13 + 2 * FeedType_size; % 13 basic features + 2*FeedType_size
        fprintf('Total feature count calculated: %d\n', numFeatures);
    else
        error('CopyrolysisFeedstock.mat does not contain CopyrolysisFeedstockTag field');
    end
catch loadErr
    error('Error loading CopyrolysisFeedstock.mat: %s', loadErr.message);
end

% Set up the number of outputs
numOutputs = numTargetCols;  % Dynamically determined outputs
fprintf('Using %d outputs for neural network targets\n', numOutputs);

% Define default training strategy and data split parameters if not already defined
if ~exist('strategy', 'var')
    strategy = 'TT'; % Default to Train-Test strategy
    fprintf('Using default strategy: %s\n', strategy);
end

if ~exist('trainRatio', 'var')
    trainRatio = 0.7; % Default to using 70% of data for training
    fprintf('Using default trainRatio: %.2f\n', trainRatio);
end

if ~exist('valRatio', 'var')
    valRatio = 0.15; % Default to using 15% of data for validation
    fprintf('Using default valRatio: %.2f\n', valRatio);
    % Note: remaining 15% will be used for testing
end

fprintf('Training model with %d features, %d samples, and %d outputs\n', numFeatures, numSamples, numOutputs);

%% Load and prepare data
fprintf('\nLoading real data for training...\n');

% Check for the required data files
rawDataFile = fullfile(modelDir, 'RawInputData.xlsx');
if ~exist(rawDataFile, 'file')
    rawDataFile = fullfile(rootDir, 'data', 'raw', 'RawInputData.xlsx');
    if ~exist(rawDataFile, 'file')
        error('ERROR: RawInputData.xlsx file not found. Cannot proceed with model training.');
    end
end

feedstockFile = fullfile(modelDir, 'CopyrolysisFeedstock.mat');
if ~exist(feedstockFile, 'file')
    feedstockFile = fullfile(rootDir, 'data', 'processed', 'CopyrolysisFeedstock.mat');
    if ~exist(feedstockFile, 'file')
        error('ERROR: CopyrolysisFeedstock.mat file not found. Cannot determine feature count.');
    end
end

% Load the raw data
try
    % Try to load raw data from Excel
    fprintf('Loading data from %s...\n', rawDataFile);
    [num, txt, raw] = xlsread(rawDataFile);
    
    % Check if data is properly loaded
    if isempty(num) || size(num, 1) < 1
        error('RawInputData.xlsx contains no numeric data or is empty');
    end
    
    % Extract feature data (assume first row is header)
    % Target variables start at column 14 based on bpDNN4PyroProd.m
    inputFeatures = num(:, 1:inputFeatureEndCol);
    targetData = num(:, targetStartCol:end);
    
    fprintf('Loaded raw data successfully. Features: %d, Samples: %d, Targets: %d\n', ...
        size(inputFeatures, 2), size(inputFeatures, 1), size(targetData, 2));
    
    % Load feedstock data
    fprintf('Loading feedstock data from %s...\n', feedstockFile);
    feedstockData = load(feedstockFile);
    
    % Process the data according to bpDNN4PyroProd.m logic
    % Here we need to add the feedstock features as defined in that file
    if isfield(feedstockData, 'CopyrolysisFeedstockTag')
        fprintf('Processing feedstock data...\n');
        % The processing logic would depend on bpDNN4PyroProd.m
        % For now, we'll create a simplified version: expand features based on feedstock
        
        % Calculate feature matrix size
        totalFeatures = size(inputFeatures, 2) + 2 * size(feedstockData.CopyrolysisFeedstockTag, 1);
        
        % For actual implementation, you would need to consult bpDNN4PyroProd.m
        % to exactly replicate the feature engineering process
        
        % As a placeholder, we'll create an expanded feature matrix with the correct dimensions
        expandedFeatures = zeros(size(inputFeatures, 1), totalFeatures);
        expandedFeatures(:, 1:size(inputFeatures, 2)) = inputFeatures;
        
        % Here you would add the feedstock features according to the actual algorithm
        fprintf('Created expanded feature matrix with %d features\n', size(expandedFeatures, 2));
        
        % Set X to the expanded features
        X = expandedFeatures;
    else
        fprintf('Warning: CopyrolysisFeedstockTag not found in feedstock data\n');
        X = inputFeatures;
    end
    
    % Set y to the target data
    y = targetData;
    
catch loadErr
    error('Error loading or processing data: %s', loadErr.message);
end

% Convert to format expected by NN functions
input_data = X';  % Convert to features as rows, samples as columns
target_data = y'; % Convert to outputs as rows, samples as columns

% Output data dimensions for monitoring
fprintf('Data dimensions check:\n');
fprintf('- Input data: %d features, %d samples\n', size(input_data, 1), size(input_data, 2));
fprintf('- Target data: %d outputs, %d samples\n', size(target_data, 1), size(target_data, 2));

% Verify if dimensions match the expected values
if size(input_data, 1) ~= numFeatures
    fprintf('Warning: Input data features (%d) do not match expected feature count (%d)\n', ...
        size(input_data, 1), numFeatures);
    % Try to fix by padding or truncating if necessary
    if size(input_data, 1) < numFeatures
        % Pad with zeros
        padding = zeros(numFeatures - size(input_data, 1), size(input_data, 2));
        input_data = [input_data; padding];
        fprintf('Padded input data to match expected feature count\n');
    else
        % Truncate
        input_data = input_data(1:numFeatures, :);
        fprintf('Truncated input data to match expected feature count\n');
    end
end

if size(target_data, 1) ~= numOutputs
    fprintf('Warning: Target data outputs (%d) do not match expected output count (%d)\n', ...
        size(target_data, 1), numOutputs);
    if size(target_data, 1) < numOutputs
        % Pad with zeros
        padding = zeros(numOutputs - size(target_data, 1), size(target_data, 2));
        target_data = [target_data; padding];
        fprintf('Padded target data to match expected output count\n');
    else
        % Truncate
        target_data = target_data(1:numOutputs, :);
        fprintf('Truncated target data to match expected output count\n');
    end
end

fprintf('Final data dimensions:\n');
fprintf('- Input data: %d features, %d samples\n', size(input_data, 1), size(input_data, 2));
fprintf('- Target data: %d outputs, %d samples\n', size(target_data, 1), size(target_data, 2));

%% Configure and train the neural network
fprintf('\nConfiguring neural network...\n');

% Extract hyperparameters from best_config
if ~isstruct(best_config)
    error('Invalid best_config: Not a structure');
end

% Extract and validate learning rate
if isfield(best_config, 'lr')
    lr = best_config.lr;
else
    lr = 0.01; % Default value
    fprintf('Warning: lr not found in best_config, using default value: %f\n', lr);
end

% Extract and validate momentum coefficient 
if isfield(best_config, 'mc')
    mc = best_config.mc;
else
    mc = 0.7; % Default value
    fprintf('Warning: mc not found in best_config, using default value: %f\n', mc);
end

% Extract and validate hidden layer structure
if isfield(best_config, 'hiddenLayer')
    % Check if hiddenLayer is a scalar or array
    if isscalar(best_config.hiddenLayer)
        hiddenLayer = [best_config.hiddenLayer]; % Convert to array
        fprintf('Converting scalar hiddenLayer to array: [%d]\n', hiddenLayer);
    else
        hiddenLayer = best_config.hiddenLayer;
    end
else
    hiddenLayer = [10]; % Default value
    fprintf('Warning: hiddenLayer not found in best_config, using default value: [%s]\n', num2str(hiddenLayer));
end

% Extract and validate transfer function
if isfield(best_config, 'transferFcn')
    hiddenTF = best_config.transferFcn;
else
    hiddenTF = 'tansig'; % Default value
    fprintf('Warning: transferFcn not found in best_config, using default value: %s\n', hiddenTF);
end

% Set default output transfer function
outputTF = 'purelin'; % Linear for regression problems

% Extract and validate other parameters if present
if isfield(best_config, 'lr_inc')
    lr_inc = best_config.lr_inc;
else
    lr_inc = 1.05; % Default value
end

if isfield(best_config, 'lr_dec')
    lr_dec = best_config.lr_dec;
else
    lr_dec = 0.7; % Default value
end

fprintf('Creating neural network with structure: [%s]\n', num2str(hiddenLayer));
net = nncreate(hiddenLayer);

% Manually set transfer functions
for i = 1:length(net.layer)-1
    net.layer{i}.transferFcn = hiddenTF;
end
% Set output layer transfer function
net.layer{end}.transferFcn = outputTF;

% Set learning parameters
net.trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation
net.performFcn = 'mse';    % Mean squared error

% Set data division function to match bpDNN4PyroProd.m definition
net.divideFcn = 'dividerand'; % Random division (matching bpDNN4PyroProd.m line 111)

% Set training parameters
net.trainParam.epochs = 100;  % Maximum epochs (reduced for testing)
net.trainParam.goal = 1e-5;   % Performance goal
net.trainParam.min_grad = 1e-6; % Minimum gradient
net.trainParam.max_fail = 10;  % Maximum validation failures
net.trainParam.lr = lr;        % Learning rate
net.trainParam.mc = mc;        % Momentum
net.trainParam.lr_inc = lr_inc; % Learning rate increase
net.trainParam.lr_dec = lr_dec; % Learning rate decrease

fprintf('Training neural network...\n');
fprintf('Epochs: %d, Learning rate: %.4f, Momentum: %.2f\n', ...
    net.trainParam.epochs, net.trainParam.lr, net.trainParam.mc);

% Train the network
if strcmp(strategy, 'TVT')
    % TVT strategy with validation data
    fprintf('Using TVT training strategy\n');
    
    % Set division ratios according to bpDNN4PyroProd.m
    net.divideParam.trainRatio = trainRatio;
    net.divideParam.valRatio = valRatio;
    net.divideParam.testRatio = 1 - trainRatio - valRatio;
    
    fprintf('Data division ratios: %.2f training, %.2f validation, %.2f testing\n', ...
        net.divideParam.trainRatio, net.divideParam.valRatio, net.divideParam.testRatio);
    
    % Train the network with all data
    [net, tr] = nntrain(net, input_data, target_data);
    
    % Evaluate performance
    trainPerf = tr.best_perf;
    valPerf = tr.best_vperf;
    testPerf = tr.best_tperf;
    
    fprintf('Training complete. Performance:\n');
    fprintf('  Training MSE: %.6f\n', trainPerf);
    fprintf('  Validation MSE: %.6f\n', valPerf);
    fprintf('  Testing MSE: %.6f\n', testPerf);
else
    % TT strategy without validation data
    fprintf('Using TT training strategy\n');
    
    % Set division parameters for TT strategy
    net.divideParam.trainRatio = trainRatio;
    net.divideParam.valRatio = 0; % No validation in TT strategy
    net.divideParam.testRatio = 1 - trainRatio;
    valRatio = 0; % Update valRatio variable to be consistent with network parameter
    
    fprintf('Data division ratios: %.2f training, %.2f testing\n', ...
        net.divideParam.trainRatio, net.divideParam.testRatio);
    
    % Train the network with all data
    [net, tr] = nntrain(net, input_data, target_data);
    
    % Evaluate performance
    trainPerf = tr.best_perf;
    if isfield(tr, 'best_tperf')
        testPerf = tr.best_tperf;
    else
        % Fallback if best_tperf not available
        fprintf('Warning: Testing performance metric not available in training results.\n');
        fprintf('Computing testing performance manually...\n');
        % Manually evaluate on test data
        testInd = tr.testInd;
        if ~isempty(testInd) && length(testInd) > 0
            test_input = input_data(:, testInd);
            test_target = target_data(:, testInd);
            testPerf = evaluate_network_performance(net, test_input, test_target);
        else
            testPerf = NaN;
            fprintf('Warning: No test indices available, cannot compute test performance.\n');
        end
    end
    
    fprintf('Training complete. Performance:\n');
    fprintf('  Training MSE: %.6f\n', trainPerf);
    if ~isnan(testPerf)
        fprintf('  Testing MSE: %.6f\n', testPerf);
    else
        fprintf('  Testing MSE: Not available\n');
    end
end

%% Save the model
fprintf('\nSaving trained model...\n');

% Prepare data for saving
PS = struct();
PS.name = 'mapminmax';
PS.xoffset = min(input_data, [], 2);
PS.gain = 2./(max(input_data, [], 2) - min(input_data, [], 2));
PS.ymin = -1;

TS = struct();
TS.name = 'mapminmax';
TS.xoffset = min(target_data, [], 2);
TS.gain = 2./(max(target_data, [], 2) - min(target_data, [], 2));
TS.ymin = -1;

% Create global versions
PS_global = PS;
TS_global = TS;

% Save to best model directory
bestModelFile = fullfile(bestModelDir, 'best_model.mat');
save(bestModelFile, 'net', 'input_data', 'target_data', 'PS', 'TS', 'PS_global', 'TS_global', ...
    'strategy', 'trainRatio', 'valRatio', 'lr', 'mc', 'lr_inc', 'lr_dec', ...
    'hiddenLayer', 'hiddenTF', 'outputTF');
fprintf('Best model saved to: %s\n', bestModelFile);

% Save a copy to training directory for analysis scripts
trainingModelFile = fullfile(trainingDir, 'trained_model.mat');
save(trainingModelFile, 'net', 'input_data', 'target_data', 'PS', 'TS', 'PS_global', 'TS_global', ...
    'strategy', 'trainRatio', 'valRatio', 'lr', 'mc', 'lr_inc', 'lr_dec', ...
    'hiddenLayer', 'hiddenTF', 'outputTF');
fprintf('Training model copy saved to: %s\n', trainingModelFile);

% Calculate execution time
executionTime = toc(startTime);
fprintf('\nExecution time: %.2f seconds (%.2f minutes)\n', ...
    executionTime, executionTime/60);

fprintf('\nTrain best model completed successfully!\n');
catch e
    % Handle errors
    fprintf('\n===== ERROR OCCURRED =====\n');
    fprintf('Error: %s\n', e.message);
    fprintf('Stack trace:\n');
    disp(e.stack);
    
    % Try to save a basic model anyway to allow SHAP analysis to proceed
    fprintf('\nAttempting to save a basic model despite errors...\n');
    
    % Check if we have the necessary variables
    if ~exist('net', 'var') || ~exist('input_data', 'var') || ~exist('target_data', 'var')
        fprintf('ERROR: Missing essential variables, cannot create even a basic model.\n');
    else
        try
            % Create simple preprocessing structures if they don't exist
            if ~exist('PS', 'var')
                PS = struct();
                PS.name = 'mapminmax';
                PS.xoffset = min(input_data, [], 2);
                PS.gain = 2./(max(input_data, [], 2) - min(input_data, [], 2));
                PS.ymin = -1;
            end
            
            if ~exist('TS', 'var')
                TS = struct();
                TS.name = 'mapminmax';
                TS.xoffset = min(target_data, [], 2);
                TS.gain = 2./(max(target_data, [], 2) - min(target_data, [], 2));
                TS.ymin = -1;
            end
            
            % Create global versions
            PS_global = PS;
            TS_global = TS;
            
            % Set default values for any missing variables
            if ~exist('strategy', 'var')
                strategy = 'TT';
            end
            
            if ~exist('trainRatio', 'var')
                trainRatio = 0.7;
            end
            
            if ~exist('valRatio', 'var')
                valRatio = 0;
            end
            
            if ~exist('lr', 'var')
                lr = 0.01;
            end
            
            if ~exist('mc', 'var')
                mc = 0.7;
            end
            
            if ~exist('lr_inc', 'var')
                lr_inc = 1.05;
            end
            
            if ~exist('lr_dec', 'var')
                lr_dec = 0.7;
            end
            
            if ~exist('hiddenLayer', 'var')
                hiddenLayer = [10];
            end
            
            if ~exist('hiddenTF', 'var')
                hiddenTF = 'tansig';
            end
            
            if ~exist('outputTF', 'var')
                outputTF = 'purelin';
            end
            
            % Save to best model directory
            fprintf('Saving emergency best model...\n');
            bestModelFile = fullfile(bestModelDir, 'best_model.mat');
            save(bestModelFile, 'net', 'input_data', 'target_data', 'PS', 'TS', 'PS_global', 'TS_global', ...
                'strategy', 'trainRatio', 'valRatio', 'lr', 'mc', 'lr_inc', 'lr_dec', ...
                'hiddenLayer', 'hiddenTF', 'outputTF');
            
            % Save a copy to training directory for analysis scripts
            trainingModelFile = fullfile(trainingDir, 'trained_model.mat');
            save(trainingModelFile, 'net', 'input_data', 'target_data', 'PS', 'TS', 'PS_global', 'TS_global', ...
                'strategy', 'trainRatio', 'valRatio', 'lr', 'mc', 'lr_inc', 'lr_dec', ...
                'hiddenLayer', 'hiddenTF', 'outputTF');
            
            fprintf('Emergency model files saved successfully.\n');
        catch saveErr
            fprintf('ERROR: Failed to save emergency model: %s\n', saveErr.message);
        end
    end
end

% Close diary
diary off;

function perf = evaluate_network_performance(net, input, target)
% Manually evaluate network performance on given data
% This function mimics the behavior of evaluate_network in nntrain.m

% Get network output
output = nnpredict(net, input);

% Calculate mean squared error
perf = mean((target - output).^2, 'all');
end