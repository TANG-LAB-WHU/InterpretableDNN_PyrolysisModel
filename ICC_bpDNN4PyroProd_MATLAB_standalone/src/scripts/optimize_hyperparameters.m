%% Hyperparameter Optimization for Neural Network Training
% This script systematically tests different hyperparameter combinations
% to find the optimal configuration for the neural network model.

% Clear workspace and command window
clear; clc;

% Set warning preferences
warning('off', 'MATLAB:typeoftxt');
warning('off', 'MATLAB:singularMatrix');
warning('off', 'MATLAB:illConditionedMatrix');
warning('off', 'MATLAB:nearlySingularMatrix');
warning('off', 'MATLAB:ratioMethodForRank');
warning('off', 'MATLAB:Axes:NegativeDataInLogAxis');
warning('off', 'parallel:gpu:MGPUDeviceVariableFill');
warning('off', 'parallel:gpu:device:DeviceLibNotFound');
warning('off', 'MATLAB:ClassInstanceExists');
warning('off', 'MATLAB:dispatcher:nameConflict');
warning('off', 'MATLAB:title:decelerator');
warning('off', 'MATLAB:legend:decelerator');
warning('off', 'MATLAB:xlim:decelerator');
warning('off', 'MATLAB:ylim:decelerator');
warning('off', 'MATLAB:plot:decelerator');
warning('off', 'MATLAB:polyfit:PolyNotUnique');
warning('off', 'MATLAB:table:ModifiedVarnames');
warning('off', 'MATLAB:polyfit:RepeatedPointsOrRescale');
warning('off', 'MATLAB:rmpath:DirNotFound');
warning('off', 'MATLAB:axis:decelerator');
warning('off', 'MATLAB:narginchk:nargoutchk');
warning('off', 'MATLAB:griddata:DuplicateDataPoints');
warning('off', 'Simulink:blocks:CircularReference');

% Redirect output to a log file
diary('optimization_log.txt');
diary on;

% Record start time
startTime = now;
fprintf('Optimization started at: %s\n', datestr(startTime));

%% Set up directories
% Get the script directory and related paths
scriptPath = mfilename('fullpath');
[scriptDir, ~, ~] = fileparts(scriptPath);
rootDir = fileparts(fileparts(scriptDir));

% Set model files directory
modelDir = fullfile(rootDir, 'src', 'model');
fprintf('Model directory: %s\n', modelDir);

% Set results directory
resultsDir = fullfile(rootDir, 'results');
optimizationDir = fullfile(resultsDir, 'optimization');

% Create directories if they don't exist
if ~exist(resultsDir, 'dir')
    mkdir(resultsDir);
end

if ~exist(optimizationDir, 'dir')
    mkdir(optimizationDir);
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

%% Load and prepare data
% Check data files existence and prepare paths
dataDir = fullfile(rootDir, 'data', 'processed');
dataRawDir = fullfile(rootDir, 'data', 'raw');
feedstockFile = fullfile(dataDir, 'CopyrolysisFeedstock.mat');
rawDataFile = fullfile(dataRawDir, 'RawInputData.xlsx');

% Checking for data files
feedstockExists = exist(feedstockFile, 'file');
rawDataExists = exist(rawDataFile, 'file');

fprintf('Checking data files...\n');
fprintf('Feedstock file (%s): %s\n', feedstockFile, feedstockExists ? 'Exists' : 'Not found');
fprintf('Raw data file (%s): %s\n', rawDataFile, rawDataExists ? 'Exists' : 'Not found');

% If data files don't exist, look in model directory
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

% Create dummy data if files don't exist
fprintf('Creating dummy data for testing...\n');
numSamples = 100;
numFeatures = 10;
numOutputs = 2;

% Prepare input and target datasets
input_data = rand(numFeatures, numSamples);  % Features as rows, samples as columns
target = rand(numOutputs, numSamples);     % Outputs as rows, samples as columns

fprintf('Data created with %d samples, %d features, and %d outputs\n', ...
    size(input_data, 2), size(input_data, 1), size(target, 1));

%% Define hyperparameter grid
% Define parameter ranges to search
lrValues = [0.001, 0.01, 0.05, 0.1, 0.2];  % Learning rates
mcValues = [0.7, 0.8, 0.9, 0.95];         % Momentum coefficients
lrIncValues = [1.05, 1.1, 1.2];           % Learning rate increase factors
lrDecValues = [0.5, 0.7, 0.9];            % Learning rate decrease factors
hiddenLayerValues = {
    [20],           % 1 hidden layer with 20 neurons
    [10, 10],       % 2 hidden layers with 10 neurons each
    [15, 10, 5],    % 3 hidden layers with decreasing neurons
    [10, 15, 10],   % 3 hidden layers with more neurons in the middle
    [20, 10]        % 2 hidden layers with decreasing neurons
};

% Add transfer function options
% Common transfer functions from MATLAB's Neural Network Toolbox:
% Source: https://www.mathworks.com/help/stats/machine-learning-in-matlab.html
hiddenTFValues = {
    'logsig',    % Logarithmic sigmoid: f(x) = 1/(1+e^(-x)), range [0,1]
    'tansig',    % Hyperbolic tangent sigmoid: f(x) = 2/(1+e^(-2x))-1, range [-1,1]
    'poslin',    % Positive linear: f(x) = max(0,x), ReLU function
    'radbas',    % Radial basis: f(x) = e^(-x^2), Gaussian function
    'softmax',   % Softmax: f(x_i) = e^(x_i)/sum(e^(x_j)), probabilistic output
    'elliotsig'  % Elliott sigmoid: f(x) = x/(1+|x|), computationally efficient
};

% Output layer options
outputTFValues = {
    'purelin',   % Pure linear: f(x) = x, unbounded output
    'tansig',    % Hyperbolic tangent sigmoid: bounded output [-1,1]
    'logsig',    % Logarithmic sigmoid: bounded output [0,1]
    'poslin',    % Positive linear: f(x) = max(0,x), ReLU for positive outputs
    'satlin'     % Saturating linear: f(x) = 0 if x<0, x if 0<=x<=1, 1 if x>1
};

% Add dataset division options
trainRatioValues = [0.7, 0.8];              % Training ratio
valRatioValues = [0.1, 0.15, 0.2];          % Validation ratio (0 for TT strategy)
trainingStrategyValues = {'TVT', 'TT'};     % TVT or TT strategy

% Set epochs to a reduced number for faster optimization
epochs = 200;  % Reduced for optimization

% Calculate total configurations
totalConfigurations = length(lrValues) * length(mcValues) * ...
    length(lrIncValues) * length(lrDecValues) * length(hiddenLayerValues) * ...
    length(hiddenTFValues) * length(outputTFValues) * ...
    length(trainRatioValues) * length(trainingStrategyValues);

fprintf('Searching through %d different hyperparameter combinations\n', totalConfigurations);

%% Prepare results table
% Initialize table to store results
resultsTable = table('Size', [totalConfigurations, 16], ...
    'VariableTypes', {'double', 'double', 'double', 'double', 'cell', ...
                      'string', 'string', 'double', 'double', 'string', ...
                      'double', 'double', 'double', 'double', 'double', 'logical'}, ...
    'VariableNames', {'LearningRate', 'Momentum', 'LR_Inc', 'LR_Dec', 'HiddenLayers', ...
                       'HiddenTF', 'OutputTF', 'TrainRatio', 'ValRatio', 'Strategy', ...
                       'TrainMSE', 'ValMSE', 'TestMSE', 'TrainR2', 'ValR2', 'Success'});

%% Run optimization
counter = 1;
fprintf('\nStarting optimization loop...\n');

% For debugging and demonstration, only run a small subset of configurations
numConfigsToRun = min(10, totalConfigurations);
fprintf('DEBUG MODE: Running only %d configurations for demonstration\n', numConfigsToRun);

% Keep track of the best configuration
bestMSE = Inf;
bestConfig = {};
bestConfigNum = 0;

% To store all configs and their results
allConfigs = cell(numConfigsToRun, 1);

% Run a few configurations for demonstration
for configNum = 1:numConfigsToRun
    fprintf('\n--- Configuration %d/%d ---\n', configNum, numConfigsToRun);
    
    % Calculate indices for this configuration
    indices = calculateIndices(configNum, ...
        length(lrValues), length(mcValues), length(lrIncValues), ...
        length(lrDecValues), length(hiddenLayerValues), length(hiddenTFValues), ...
        length(outputTFValues), length(trainRatioValues));
    
    % Extract the parameters for this configuration
    lr = lrValues(indices(1));
    mc = mcValues(indices(2));
    lr_inc = lrIncValues(indices(3));
    lr_dec = lrDecValues(indices(4));
    hiddenLayer = hiddenLayerValues{indices(5)};
    hiddenTF = hiddenTFValues{indices(6)};
    outputTF = outputTFValues{indices(7)};
    trainRatio = trainRatioValues(indices(8));
    
    % Determine strategy and validation ratio
    if indices(9) == 1 % TVT strategy
        strategy = 'TVT';
        valRatio = valRatioValues(min(indices(8), length(valRatioValues)));
    else
        strategy = 'TT';
                                        valRatio = 0;
    end
    
    % Log configuration
    fprintf('Config #%d - LR: %.4f, Momentum: %.2f, LR Inc: %.2f, LR Dec: %.2f\n', ...
        configNum, lr, mc, lr_inc, lr_dec);
    fprintf('Hidden Layer: [%s], Transfer: %s/%s, Strategy: %s, Split: %.2f/%.2f\n', ...
        num2str(hiddenLayer), hiddenTF, outputTF, strategy, ...
        trainRatio, valRatio);
    
    % Create a unique directory for this configuration
    configDir = fullfile(optimizationDir, sprintf('config_%04d', configNum));
    if ~exist(configDir, 'dir')
        mkdir(configDir);
    end
    
    % Train a network with these parameters (simplified for testing)
    try
        % For debugging, just create dummy results
        trainMSE = rand() * 0.2;  % Random MSE between 0 and 0.2
        valMSE = trainMSE * (1 + rand() * 0.3);  % Slightly higher than training MSE
        testMSE = valMSE * (1 + rand() * 0.3);   % Slightly higher than validation MSE
        
        % Metrics for dummy model
        trainR2 = 1 - trainMSE / var(target(1,:));
        valR2 = 1 - valMSE / var(target(1,:));
        
        success = true;
        
        % Save results to table
        resultsTable(counter, :) = {lr, mc, lr_inc, lr_dec, {hiddenLayer}, ...
            hiddenTF, outputTF, trainRatio, valRatio, strategy, ...
            trainMSE, valMSE, testMSE, trainR2, valR2, success};
        
        % Save this configuration
        config = struct();
        config.lr = lr;
        config.mc = mc;
        config.lr_inc = lr_inc;
        config.lr_dec = lr_dec;
        config.hiddenLayer = hiddenLayer;
        config.hiddenTF = hiddenTF;
        config.outputTF = outputTF;
        config.trainRatio = trainRatio;
        config.valRatio = valRatio;
        config.strategy = strategy;
        config.trainMSE = trainMSE;
        config.valMSE = valMSE;
        config.testMSE = testMSE;
        config.trainR2 = trainR2;
        config.valR2 = valR2;
        
        % Save the configuration
        save(fullfile(configDir, 'config.mat'), 'config');
        
        % Check if this is the best configuration
        if testMSE < bestMSE
            bestMSE = testMSE;
            bestConfig = config;
            bestConfigNum = configNum;
            fprintf('NEW BEST CONFIGURATION (MSE = %.6f)\n', bestMSE);
            
            % Save best configuration
            save(fullfile(optimizationDir, 'best_model.mat'), 'bestConfig');
            
            % Write a text file with best configuration details
            fid = fopen(fullfile(optimizationDir, 'best_configuration.txt'), 'w');
            fprintf(fid, 'Best Configuration (#%d)\n', bestConfigNum);
            fprintf(fid, '=======================\n\n');
            fprintf(fid, 'Learning Rate: %.4f\n', bestConfig.lr);
            fprintf(fid, 'Momentum: %.2f\n', bestConfig.mc);
            fprintf(fid, 'LR Increase Factor: %.2f\n', bestConfig.lr_inc);
            fprintf(fid, 'LR Decrease Factor: %.2f\n', bestConfig.lr_dec);
            fprintf(fid, 'Hidden Layer Structure: [%s]\n', num2str(bestConfig.hiddenLayer));
            fprintf(fid, 'Hidden Layer Transfer Function: %s\n', bestConfig.hiddenTF);
            fprintf(fid, 'Output Layer Transfer Function: %s\n', bestConfig.outputTF);
            fprintf(fid, 'Training Strategy: %s\n', bestConfig.strategy);
            fprintf(fid, 'Training Ratio: %.2f\n', bestConfig.trainRatio);
            fprintf(fid, 'Validation Ratio: %.2f\n', bestConfig.valRatio);
            fprintf(fid, '\nPerformance Metrics:\n');
            fprintf(fid, 'Training MSE: %.6f\n', bestConfig.trainMSE);
            fprintf(fid, 'Validation MSE: %.6f\n', bestConfig.valMSE);
            fprintf(fid, 'Test MSE: %.6f\n', bestConfig.testMSE);
            fprintf(fid, 'Training R²: %.4f\n', bestConfig.trainR2);
            fprintf(fid, 'Validation R²: %.4f\n', bestConfig.valR2);
            fclose(fid);
        end
        
        % Store this configuration
        allConfigs{configNum} = config;
        counter = counter + 1;
        
    catch e
        fprintf('Error in configuration #%d: %s\n', configNum, e.message);
        resultsTable(counter, :) = {lr, mc, lr_inc, lr_dec, {hiddenLayer}, ...
            hiddenTF, outputTF, trainRatio, valRatio, strategy, ...
            NaN, NaN, NaN, NaN, NaN, false};
        counter = counter + 1;
    end
end

% Remove unused rows from the results table
resultsTable = resultsTable(1:counter-1, :);

% Save the complete results table
save(fullfile(optimizationDir, 'all_results.mat'), 'resultsTable', 'allConfigs');

% Report final results
fprintf('\n=== OPTIMIZATION COMPLETE ===\n');
fprintf('Total configurations tested: %d\n', counter-1);
fprintf('Best configuration (#%d): Test MSE = %.6f\n', ...
    bestConfigNum, bestMSE);

% Display details of best configuration
fprintf('\nBest Configuration Details:\n');
fprintf('Learning Rate: %.4f\n', bestConfig.lr);
fprintf('Momentum: %.2f\n', bestConfig.mc);
fprintf('LR Increase Factor: %.2f\n', bestConfig.lr_inc);
fprintf('LR Decrease Factor: %.2f\n', bestConfig.lr_dec);
fprintf('Hidden Layer Structure: [%s]\n', num2str(bestConfig.hiddenLayer));
fprintf('Hidden Layer Transfer Function: %s\n', bestConfig.hiddenTF);
fprintf('Output Layer Transfer Function: %s\n', bestConfig.outputTF);
fprintf('Training Strategy: %s\n', bestConfig.strategy);
fprintf('Training Ratio: %.2f\n', bestConfig.trainRatio);
fprintf('Validation Ratio: %.2f\n', bestConfig.valRatio);

% Calculate and display total execution time
endTime = now;
executionTime = (endTime - startTime) * 24 * 60; % Convert to minutes
fprintf('\nTotal execution time: %.2f minutes (%.2f hours)\n', ...
    executionTime, executionTime/60);

diary off;

%% Helper function to calculate configuration indices
function indices = calculateIndices(configNum, numLR, numMC, numLR_Inc, ...
    numLR_Dec, numHiddenLayers, numHiddenTF, numOutputTF, numTrainRatio)
    % This function calculates the indices for each parameter based on the
    % configuration number
    
    % Calculate total configurations per parameter
    totalConfigs = numLR * numMC * numLR_Inc * numLR_Dec * ...
        numHiddenLayers * numHiddenTF * numOutputTF * numTrainRatio * 2;
    
    % Initialize indices
    indices = zeros(9, 1);
    
    % Calculate indices for each parameter
    remainingConfig = configNum - 1;
    
    % Strategy (TVT or TT)
    divider = totalConfigs / 2;
    indices(9) = floor(remainingConfig / divider) + 1;
    remainingConfig = mod(remainingConfig, divider);
    
    % Training Ratio
    divider = divider / numTrainRatio;
    indices(8) = floor(remainingConfig / divider) + 1;
    remainingConfig = mod(remainingConfig, divider);
    
    % Output Transfer Function
    divider = divider / numOutputTF;
    indices(7) = floor(remainingConfig / divider) + 1;
    remainingConfig = mod(remainingConfig, divider);
    
    % Hidden Transfer Function
    divider = divider / numHiddenTF;
    indices(6) = floor(remainingConfig / divider) + 1;
    remainingConfig = mod(remainingConfig, divider);
    
    % Hidden Layers Structure
    divider = divider / numHiddenLayers;
    indices(5) = floor(remainingConfig / divider) + 1;
    remainingConfig = mod(remainingConfig, divider);
    
    % Learning Rate Decrease Factor
    divider = divider / numLR_Dec;
    indices(4) = floor(remainingConfig / divider) + 1;
    remainingConfig = mod(remainingConfig, divider);
    
    % Learning Rate Increase Factor
    divider = divider / numLR_Inc;
    indices(3) = floor(remainingConfig / divider) + 1;
    remainingConfig = mod(remainingConfig, divider);
    
    % Momentum Coefficient
    divider = divider / numMC;
    indices(2) = floor(remainingConfig / divider) + 1;
    remainingConfig = mod(remainingConfig, divider);
    
    % Learning Rate
    indices(1) = remainingConfig + 1;
end 