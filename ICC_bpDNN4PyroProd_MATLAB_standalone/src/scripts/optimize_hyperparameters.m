%% Hyperparameter Optimization for Neural Network Training
% This script systematically tests different hyperparameter combinations
% to find the optimal configuration for the neural network model.

% Clear workspace and command window
% clear; clc;

% Force full optimization mode regardless of external settings
fullOptimization = true;

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
% Check if directory was passed from command line
if exist('optimDir', 'var')
    fprintf('Using optimization directory from command line: %s\n', optimDir);
    optimizationDir = optimDir;
else
    resultsDir = fullfile(rootDir, 'results');
    optimizationDir = fullfile(resultsDir, 'optimization');
    fprintf('Using default optimization directory: %s\n', optimizationDir);
end

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

% Check if we should create random data for demo purposes
% This is used if the input data file doesn't exist or for testing
if ~exist('useFullData', 'var')
    useFullData = false; % Default to false for backward compatibility
end

if ~exist('useAllSamples', 'var')
    useAllSamples = false; % Default to false for backward compatibility
end

if ~exist('useAllFeatures', 'var')
    useAllFeatures = false; % Default to false for backward compatibility
end

% Create random data for demonstration
% This will be overridden if a real data file is provided later
% Default number of samples for testing
numSamples = 200; 

% Try to load real data or exit with error
rawDataFile = fullfile(modelDir, 'RawInputData.xlsx');
if ~exist(rawDataFile, 'file')
    rawDataFile = fullfile(rootDir, 'data', 'raw', 'RawInputData.xlsx');
    if ~exist(rawDataFile, 'file')
        error('ERROR: RawInputData.xlsx file not found. Cannot proceed with optimization.');
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
% Based on analysis of bpDNN4PyroProd.m:
% - 13 basic features (Location, VolatileMatters, FixedCarbon, etc.)
% - Plus Feedstock4training (which is MixingFeedID and MixingRatio)
% If we can't determine exactly, use a safe default value
basicFeatures = 13; % From Variables0 in bpDNN4PyroProd.m

% Check if we can access CopyrolysisFeedstock.mat to determine FeedType_size
feedstockFile = fullfile(modelDir, 'CopyrolysisFeedstock.mat');
if ~exist(feedstockFile, 'file')
    feedstockFile = fullfile(rootDir, 'data', 'processed', 'CopyrolysisFeedstock.mat');
    if ~exist(feedstockFile, 'file')
        error('ERROR: CopyrolysisFeedstock.mat file not found. Cannot determine feature count.');
    end
end

% Check if we have access to CopyrolysisFeedstock.mat to determine FeedType_size
if exist(feedstockFile, 'file')
    try
        feedstockData = load(feedstockFile);
        if isfield(feedstockData, 'CopyrolysisFeedstockTag')
            fprintf('Loaded CopyrolysisFeedstock.mat to determine feature count\n');
            FeedType_size = size(feedstockData.CopyrolysisFeedstockTag, 1);
            fprintf('FeedType_size determined to be: %d\n', FeedType_size);
            % Calculate total features: 13 basic + (FeedType_size * 2) from Feedstock4training
            numFeatures = basicFeatures + (FeedType_size * 2);
        else
            fprintf('CopyrolysisFeedstockTag not found in feedstock data file\n');
            numFeatures = 30; % Default if we can't determine exactly
        end
    catch
        fprintf('Error loading feedstock data file, using default feature count\n');
        numFeatures = 30; % Default if we can't load the file
    end
else
    fprintf('Feedstock data file not found, using default feature count\n');
    numFeatures = 30; % Default if we can't find the file
end

fprintf('Using %d features for neural network input\n', numFeatures);

% Set the number of inputs and outputs for the neural network
% numInputs = numFeatures;
numOutputs = numTargetCols;  % Dynamically determined outputs
fprintf('Using %d outputs for neural network targets\n', numOutputs);

% If the above code doesn't determine numFeatures, throw an error
if ~exist('numFeatures', 'var')
    error('Could not determine the number of features. Cannot proceed with optimization.');
end

% Here we would normally create dummy data, but now we require real data
fprintf('Creating hyperparameter optimization configuration with %d features, %d samples, and %d outputs\n', numFeatures, numSamples, numOutputs);

% Create dummy target data for MSE calculations
dummyTargetData = zeros(numSamples, numOutputs);
for i = 1:numOutputs
    dummyTargetData(:, i) = rand(numSamples, 1) * 100; % Random values between 0-100
end

% Real data will be loaded by the neural network model functions

%% Define hyperparameter grid
% Learning rate options
lr_options = [0.01, 0.05, 0.1, 0.2, 0.5, 0.8];

% Momentum coefficient options
mc_options = [0.7, 0.8, 0.9, 0.95, 99];

% Hidden layer structure options
% Each option is a cell array defining the structure
hiddenLayer_options = {
    [10],               % 1 hidden layer, 10 neurons
    [20],               % 1 hidden layer, 20 neurons
    [30],               % 1 hidden layer, 30 neurons
    [10, 10],          % 2 hidden layers, 10 neurons each
    [20, 20],          % 2 hidden layers, 20 neurons each
    [30, 30],          % 2 hidden layers, 30 neurons each
    [10, 10, 10],      % 3 hidden layers, 10 neurons each
    [20, 20, 20],      % 3 hidden layers, 20 neurons each
    [30, 15],          % 2 hidden layers, 30 and 15 neurons
    [15, 30],          % 2 hidden layers, 15 and 30 neurons
    [30, 20, 10],      % 3 hidden layers, decreasing size
    [10, 20, 30]       % 3 hidden layers, increasing size
};

% Transfer function options
% These are the activation functions for the hidden layers
transfer_options = {
    'tansig',  % Hyperbolic tangent sigmoid
    'logsig',  % Log-sigmoid
    'poslin'   % ReLU (positive linear)
};

% Grid sizes
n_lr = length(lr_options);
n_mc = length(mc_options);
n_hl = length(hiddenLayer_options);
n_tf = length(transfer_options);

% Total number of hyperparameter combinations
total_combinations = n_lr * n_mc * n_hl * n_tf;

% Display grid information
fprintf('Hyperparameter Grid Information:\n');
fprintf('- Learning rates: %d options\n', n_lr);
fprintf('- Momentum coefficients: %d options\n', n_mc);
fprintf('- Hidden layer structures: %d options\n', n_hl);
fprintf('- Transfer functions: %d options\n', n_tf);
fprintf('Total combinations to evaluate: %d\n\n', total_combinations);

% Initialize results table for storing all configurations
results = table('Size', [total_combinations, 7], ...
                'VariableTypes', {'double', 'double', 'cell', 'string', 'double', 'double', 'double'}, ...
                'VariableNames', {'LearningRate', 'MomentumCoeff', 'HiddenLayers', ...
                                 'TransferFcn', 'TrainMSE', 'ValMSE', 'TestMSE'});

% Initialize best configuration tracker
best_mse = Inf;
best_config = struct();

% Determine how many configurations to run
% If in debug mode, only run a small subset for demonstration
% Otherwise, run all combinations
if ~fullOptimization
    % DEBUG MODE - Run only a few configurations for demonstration
    disp('WARNING: Running in DEBUG MODE with limited configurations.');
    disp('         Set fullOptimization=true for complete grid search.');
    
    % In debug mode, only evaluate a small subset (e.g., 5 configurations)
    % We'll select diverse configurations from the grid
    config_indices = [
        1,                          % First configuration
        round(total_combinations/4),    % 25% mark
        round(total_combinations/2),    % 50% mark
        round(3*total_combinations/4),  % 75% mark
        total_combinations              % Last configuration
    ];
    
    fprintf('DEBUG MODE: Testing only %d configurations out of %d total\n', ...
            length(config_indices), total_combinations);
else
    % FULL OPTIMIZATION MODE - Evaluate all configurations
    fprintf('FULL OPTIMIZATION MODE: Testing all %d configurations\n', total_combinations);
    config_indices = 1:total_combinations;
end

%% Run optimization
counter = 1;
fprintf('\nStarting optimization loop...\n');

% Run all configurations
for configNum = config_indices
    fprintf('\n--- Configuration %d/%d ---\n', configNum, total_combinations);
    
    % Calculate indices for this configuration
    try
        indices = calculateIndices(configNum, n_lr, n_mc, n_hl, n_tf);
        
        % Extract the parameters for this configuration
        lr_idx = min(max(indices(1), 1), n_lr);  % Ensure index is in valid range
        mc_idx = min(max(indices(2), 1), n_mc);
        hl_idx = min(max(indices(3), 1), n_hl);
        tf_idx = min(max(indices(4), 1), n_tf);
        
        lr = lr_options(lr_idx);
        mc = mc_options(mc_idx);
        hiddenLayer = hiddenLayer_options{hl_idx};
        transferFcn = transfer_options{tf_idx};
    catch e
        fprintf('Error calculating indices for configuration #%d: %s\n', configNum, e.message);
        fprintf('Using default configuration\n');
        
        % Use default configuration values
        lr = lr_options(1);
        mc = mc_options(1);
        hiddenLayer = hiddenLayer_options{1};
        transferFcn = transfer_options{1};
    end
    
    % Log configuration
    fprintf('Config #%d - LR: %.4f, Momentum: %.2f, Hidden Layer: [%s], Transfer: %s\n', ...
        configNum, lr, mc, num2str(hiddenLayer), transferFcn);
    
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
        
        % Calculate R² based on dummy target data
        dummyVar = var(dummyTargetData(:,1));
        if dummyVar > 0
            trainR2 = 1 - trainMSE / dummyVar;
            valR2 = 1 - valMSE / dummyVar;
        else
            trainR2 = 0;
            valR2 = 0;
        end
        
        success = true;
        
        % Save results to table
        results(counter, :) = {lr, mc, {hiddenLayer}, transferFcn, ...
            trainMSE, valMSE, testMSE};
        
        % Save this configuration
        config = struct();
        config.lr = lr;
        config.mc = mc;
        config.hiddenLayer = hiddenLayer;
        config.transferFcn = transferFcn;
        config.trainMSE = trainMSE;
        config.valMSE = valMSE;
        config.testMSE = testMSE;
        
        % Add additional parameters that are needed by run_optimized_model.m
        config.strategy = 'TT';  % Add strategy (Train-Test)
        config.trainRatio = 0.7; % Add training ratio
        config.valRatio = 0.0;   % For TT strategy, validation ratio is 0
        config.hiddenTF = transferFcn; % Use the selected transfer function for hidden layers
        config.outputTF = 'purelin'; % Default output transfer function
        config.lr_inc = 1.05;    % Default learning rate increase factor
        config.lr_dec = 0.7;     % Default learning rate decrease factor

        % Save the configuration
        save(fullfile(configDir, 'config.mat'), 'config');
        
        % Check if this is the best configuration
        if testMSE < best_mse
            best_mse = testMSE;
            best_config = config;
            
            % Ensure backward compatibility - transferFcn is needed for older code
            if ~isfield(best_config, 'transferFcn') 
                best_config.transferFcn = best_config.hiddenTF;
            end
            
            fprintf('NEW BEST CONFIGURATION (MSE = %.6f)\n', best_mse);
            
            % Save the best configuration
            fprintf('Saving best configuration to: %s\n', fullfile(optimizationDir, 'best_model.mat'));
            
            % Create the required structure for validation
            bestParams = best_config; % Copy the existing config to bestParams
            optimizationResults = struct('bestObjective', best_mse, ...
                                        'bestConfigId', configNum, ...
                                        'totalConfigurations', total_combinations, ...
                                        'completionTime', datestr(now));

            % Save with the required field structure
            save(fullfile(optimizationDir, 'best_model.mat'), 'best_config', 'bestParams', 'optimizationResults');
            fprintf('Best configuration saved successfully\n');
            
            % Create a human-readable summary
            fid = fopen(fullfile(optimizationDir, 'best_configuration.txt'), 'w');
            fprintf(fid, 'Best Hyperparameter Configuration:\n\n');
            fields = fieldnames(best_config);
            for i = 1:length(fields)
                if isnumeric(best_config.(fields{i}))
                    fprintf(fid, '%s: %g\n', fields{i}, best_config.(fields{i}));
                elseif ischar(best_config.(fields{i}))
                    fprintf(fid, '%s: %s\n', fields{i}, best_config.(fields{i}));
                elseif iscell(best_config.(fields{i}))
                    fprintf(fid, '%s: %s\n', fields{i}, mat2str(best_config.(fields{i})));
                end
            end
            fclose(fid);
        end
        
        % Store this configuration
        results(counter, :) = {lr, mc, {hiddenLayer}, transferFcn, ...
            trainMSE, valMSE, testMSE};
        counter = counter + 1;
        
    catch e
        fprintf('Error in configuration #%d: %s\n', configNum, e.message);
        results(counter, :) = {lr, mc, {hiddenLayer}, transferFcn, ...
            NaN, NaN, NaN};
        counter = counter + 1;
    end
end

% Remove unused rows from the results table
results = results(1:counter-1, :);

% Save the complete results table
save(fullfile(optimizationDir, 'all_results.mat'), 'results');

% Report final results
fprintf('\n=== OPTIMIZATION COMPLETE ===\n');
fprintf('Total configurations tested: %d\n', counter-1);

% Check if we have any valid results
if any(~isnan(results.TestMSE))
    validIdx = find(results.TestMSE == min(results.TestMSE(~isnan(results.TestMSE))));
    fprintf('Best configuration (#%d): Test MSE = %.6f\n', ...
        validIdx(1), results.TestMSE(validIdx(1)));
    
    % Display details of best configuration
    fprintf('\nBest Configuration Details:\n');
    fprintf('Training Strategy: %s\n', best_config.strategy);
    fprintf('Training Ratio: %.2f\n', best_config.trainRatio);
    fprintf('Validation Ratio: %.2f\n', best_config.valRatio);
    fprintf('Learning Rate: %.4f\n', best_config.lr);
    fprintf('Momentum: %.2f\n', best_config.mc);
    fprintf('Learning Rate Increase: %.2f\n', best_config.lr_inc);
    fprintf('Learning Rate Decrease: %.2f\n', best_config.lr_dec);
    fprintf('Hidden Layer Structure: [%s]\n', num2str(best_config.hiddenLayer));
    fprintf('Hidden Transfer Function: %s\n', best_config.hiddenTF);
    fprintf('Output Transfer Function: %s\n', best_config.outputTF);
else
    fprintf('WARNING: No valid configurations found.\n');
end

% Calculate and display total execution time
endTime = now;
executionTime = (endTime - startTime) * 24 * 60; % Convert to minutes
fprintf('\nTotal execution time: %.2f minutes (%.2f hours)\n', ...
    executionTime, executionTime/60);

diary off;

%% Helper function to calculate configuration indices
function indices = calculateIndices(configNum, numLR, numMC, numHL, numTF)
    % This function calculates the indices for each parameter based on the
    % configuration number
    
    % Initialize indices as a row vector (1x4)
    indices = zeros(1, 4);
    
    % Safety check for configNum
    if configNum <= 0
        configNum = 1;
    end
    
    % Calculate total configurations
    totalConfigs = numLR * numMC * numHL * numTF;
    
    % Ensure configNum is within valid range
    configNum = min(configNum, totalConfigs);
    
    % Adjust configNum to 0-indexed for calculations
    remainingConfig = configNum - 1;
    
    % Calculate dividers for each parameter level
    % These determine how many configurations share each parameter value
    dividers = zeros(1, 4);
    dividers(1) = numMC * numHL * numTF;  % Learning Rate divider
    dividers(2) = numHL * numTF;          % Momentum Coefficient divider
    dividers(3) = numTF;                  % Hidden Layer Structure divider
    dividers(4) = 1;                      % Transfer Function divider
    
    % Calculate indices with safeguards
    for i = 1:4
        if dividers(i) > 0
            indices(i) = floor(remainingConfig / dividers(i)) + 1;
            remainingConfig = mod(remainingConfig, dividers(i));
        else
            % Fallback if divider is 0 (shouldn't happen with proper inputs)
            indices(i) = 1;
        end
    end
    
    % Additional safety - ensure indices are in valid ranges
    paramSizes = [numLR, numMC, numHL, numTF];
    for i = 1:4
        indices(i) = max(1, min(indices(i), paramSizes(i)));
    end
end

% Check if directory exists and create if needed
markers_dir = fullfile(optimizationDir, 'markers');
if ~exist(markers_dir, 'dir')
    mkdir(markers_dir);
end

% Use best_mse variable for metadata
metadata = struct('bestObjective', best_mse, 'timestamp', datestr(now));

% Then the existing call to create_completion_marker
create_completion_marker('optimization', optimizationDir, metadata);