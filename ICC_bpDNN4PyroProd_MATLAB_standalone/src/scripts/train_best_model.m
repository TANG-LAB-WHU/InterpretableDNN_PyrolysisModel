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
    fprintf('Feedstock file (%s): %s\n', feedstockFile, feedstockExists ? 'Exists' : 'Not found');
    fprintf('Raw data file (%s): %s\n', rawDataFile, rawDataExists ? 'Exists' : 'Not found');
    
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

% Try to load best configuration
    bestModelMatFile = fullfile(optimizationDir, 'best_model.mat');
    if exist(bestModelMatFile, 'file')
        fprintf('Loading best model configuration from %s\n', bestModelMatFile);
        bestModel = load(bestModelMatFile);
        
        % Check if the file contains bestConfig or bestParameters
        if isfield(bestModel, 'bestConfig')
            bestConfig = bestModel.bestConfig;
        elseif isfield(bestModel, 'bestParameters')
            bestConfig = bestModel.bestParameters;
        else
            error('Best model file does not contain bestConfig or bestParameters');
        end
        
        % Extract parameters
        lr = bestConfig.lr;
        mc = bestConfig.mc;
        lr_inc = bestConfig.lr_inc;
        lr_dec = bestConfig.lr_dec;
        hiddenLayer = bestConfig.hiddenLayer;
        hiddenTF = bestConfig.hiddenTF;
        outputTF = bestConfig.outputTF;
        trainRatio = bestConfig.trainRatio;
        valRatio = bestConfig.valRatio;
        strategy = bestConfig.strategy;
        
        fprintf('Best hyperparameters loaded:\n');
        fprintf('  Learning rate: %.4f\n', lr);
        fprintf('  Momentum: %.2f\n', mc);
        fprintf('  LR increase factor: %.2f\n', lr_inc);
        fprintf('  LR decrease factor: %.2f\n', lr_dec);
        fprintf('  Hidden layer structure: [%s]\n', num2str(hiddenLayer));
        fprintf('  Hidden transfer function: %s\n', hiddenTF);
        fprintf('  Output transfer function: %s\n', outputTF);
        fprintf('  Training strategy: %s\n', strategy);
        fprintf('  Training ratio: %.2f\n', trainRatio);
        fprintf('  Validation ratio: %.2f\n', valRatio);
    else
        fprintf('Best model file not found: %s\n', bestModelMatFile);
        fprintf('Using default hyperparameters...\n');
        
        % Default parameters
    lr = 0.01;
    mc = 0.9;
    lr_inc = 1.05;
    lr_dec = 0.7;
        hiddenLayer = [10, 10];
        hiddenTF = 'tansig';
    outputTF = 'purelin';
        trainRatio = 0.7;
        valRatio = 0.15;
    strategy = 'TVT';
        
        fprintf('Default hyperparameters:\n');
        fprintf('  Learning rate: %.4f\n', lr);
        fprintf('  Momentum: %.2f\n', mc);
        fprintf('  LR increase factor: %.2f\n', lr_inc);
        fprintf('  LR decrease factor: %.2f\n', lr_dec);
        fprintf('  Hidden layer structure: [%s]\n', num2str(hiddenLayer));
        fprintf('  Hidden transfer function: %s\n', hiddenTF);
        fprintf('  Output transfer function: %s\n', outputTF);
        fprintf('  Training strategy: %s\n', strategy);
        fprintf('  Training ratio: %.2f\n', trainRatio);
        fprintf('  Validation ratio: %.2f\n', valRatio);
    end
    
    %% Load and prepare data
    fprintf('\nCreating dummy data for testing since we cannot load actual data...\n');
    
    % Create dummy data
    numFeatures = 10;
    numSamples = 200;
    numOutputs = 2;
    
    % Generate random input and target data
    input = rand(numFeatures, numSamples);  % Features x Samples
    target = rand(numOutputs, numSamples);  % Outputs x Samples
    
    % Split data according to strategy
    if strcmp(strategy, 'TVT')
        % Training-Validation-Testing split
        trainEnd = floor(trainRatio * numSamples);
        valEnd = trainEnd + floor(valRatio * numSamples);
        
        trainInput = input(:, 1:trainEnd);
        trainTarget = target(:, 1:trainEnd);
        
        valInput = input(:, trainEnd+1:valEnd);
        valTarget = target(:, trainEnd+1:valEnd);
        
        testInput = input(:, valEnd+1:end);
        testTarget = target(:, valEnd+1:end);
        
        fprintf('Data split (TVT): %d training, %d validation, %d testing samples\n', ...
            size(trainInput, 2), size(valInput, 2), size(testInput, 2));
    else
        % Training-Testing split
        trainEnd = floor(trainRatio * numSamples);
        
        trainInput = input(:, 1:trainEnd);
        trainTarget = target(:, 1:trainEnd);
        
        valInput = [];
        valTarget = [];
        
        testInput = input(:, trainEnd+1:end);
        testTarget = target(:, trainEnd+1:end);
        
        fprintf('Data split (TT): %d training, %d testing samples\n', ...
            size(trainInput, 2), size(testInput, 2));
    end
    
    %% Configure and train the neural network
    fprintf('\nConfiguring neural network...\n');
    
    % Create the neural network
    fprintf('Creating neural network with structure: [%s]\n', num2str(hiddenLayer));
    net = nncreate(hiddenLayer, hiddenTF, outputTF);
    
    % Set learning parameters
    net.trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation
    net.performFcn = 'mse';    % Mean squared error
    net.divideFcn = 'divideind'; % Divide by indices
    
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
        
        % Prepare data indices
        trainInd = 1:size(trainInput, 2);
        valInd = size(trainInput, 2) + (1:size(valInput, 2));
        testInd = size(trainInput, 2) + size(valInput, 2) + (1:size(testInput, 2));
        
        % Combine data for training
        X = [trainInput, valInput, testInput];
        T = [trainTarget, valTarget, testTarget];
        
        % Set division indices
        net.divideFcn = 'divideind';
        net.divideParam.trainInd = trainInd;
        net.divideParam.valInd = valInd;
        net.divideParam.testInd = testInd;
        
        % Train the network
        [net, tr] = nntrain(net, X, T);
        
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
        
        % Set no validation
        net.divideFcn = '';
        
        % Train the network
        [net, tr] = nntrain(net, trainInput, trainTarget);
        
        % Evaluate performance on test data
        testPerf = nneval(net, testInput, testTarget);
        
        fprintf('Training complete. Performance:\n');
        fprintf('  Training MSE: %.6f\n', tr.best_perf);
        fprintf('  Testing MSE: %.6f\n', testPerf);
    end
    
    %% Save the model
    fprintf('\nSaving trained model...\n');
    
    % Prepare data for saving
    PS = struct();
    PS.name = 'mapminmax';
    PS.xoffset = min(input, [], 2);
    PS.gain = 2./(max(input, [], 2) - min(input, [], 2));
    PS.ymin = -1;
    
    TS = struct();
    TS.name = 'mapminmax';
    TS.xoffset = min(target, [], 2);
    TS.gain = 2./(max(target, [], 2) - min(target, [], 2));
    TS.ymin = -1;
    
    % Create global versions
    PS_global = PS;
    TS_global = TS;
    
    % Save to best model directory
    bestModelFile = fullfile(bestModelDir, 'best_model.mat');
    save(bestModelFile, 'net', 'input', 'target', 'PS', 'TS', 'PS_global', 'TS_global', ...
        'strategy', 'trainRatio', 'valRatio', 'lr', 'mc', 'lr_inc', 'lr_dec', ...
        'hiddenLayer', 'hiddenTF', 'outputTF');
    fprintf('Best model saved to: %s\n', bestModelFile);
    
    % Save a copy to training directory for analysis scripts
    trainingModelFile = fullfile(trainingDir, 'Results_trained.mat');
    save(trainingModelFile, 'net', 'input', 'target', 'PS', 'TS', 'PS_global', 'TS_global', ...
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
end

% Close diary
diary off;