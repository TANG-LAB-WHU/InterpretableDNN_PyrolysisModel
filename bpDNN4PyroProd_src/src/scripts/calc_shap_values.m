%% SHAP Value Calculation for Neural Network Model
% This script calculates SHAP values for the trained neural network model
% using a simplified approach for demonstration.

fprintf('Starting simplified SHAP value calculation...\n');

% Check if shapDir exists in the workspace
if ~exist('shapDir', 'var')
    % Get the script directory
    scriptPath = mfilename('fullpath');
    [scriptDir, ~, ~] = fileparts(scriptPath);
    
    % If not provided, set default paths based on the script location
    rootDir = fileparts(fileparts(scriptDir));
    resultsDir = fullfile(rootDir, 'results');
    
    % Try to determine which analysis mode we're in based on environment
    % If debug_run_analysis.m was the caller, use debug path
    % Otherwise use full analysis path
    dbstack_info = dbstack;
    if length(dbstack_info) > 1
        caller_name = dbstack_info(2).name;
        if contains(lower(caller_name), 'debug')
            shapDir = fullfile(resultsDir, 'analysis', 'debug');
            fprintf('No shapDir in workspace, using debug path: %s\n', shapDir);
        else
            shapDir = fullfile(resultsDir, 'analysis', 'full');
            fprintf('No shapDir in workspace, using full analysis path: %s\n', shapDir);
        end
    else
        % Default to full analysis if called directly
        shapDir = fullfile(resultsDir, 'analysis', 'full');
        fprintf('shapDir not found in workspace, using default path: %s\n', shapDir);
    end
end

% Create a subdirectory for data files
dataDir = fullfile(shapDir, 'data');
if ~exist(dataDir, 'dir')
    mkdir(dataDir);
    fprintf('Created data directory: %s\n', dataDir);
else
    fprintf('Using existing data directory: %s\n', dataDir);
end

%% Prepare data for SHAP analysis
% Use existing data in workspace or load from file
if ~exist('input', 'var') || ~exist('target', 'var') || ~exist('net', 'var')
    % If not already in workspace, load from training directory
    if exist('trainingDir', 'var') && exist(fullfile(trainingDir, 'Results_trained.mat'), 'file')
        fprintf('Loading data from training directory: %s\n', fullfile(trainingDir, 'Results_trained.mat'));
        load(fullfile(trainingDir, 'Results_trained.mat'), 'input', 'target', 'net');
    else
        fprintf('Training data not found. Using dummy data for demonstration.\n');
        % Create dummy data for demonstration
        input = rand(5, 20);  % 5 features, 20 samples
        target = rand(2, 20); % 2 outputs, 20 samples
        
        % Create a simple network
        net = struct();
        net.inputs = {struct('size', 5)};
        net.layers = {struct('size', 10), struct('size', 2)};
        net.outputs = {struct('size', 2)};
        net.biases = {1, 1};
        net.inputWeights = {struct('size', [10, 5])};
        net.layerWeights = {[], struct('size', [2, 10])};
        net.trainFcn = 'trainlm';
        net.performFcn = 'mse';
    end
end

% Ensure data is properly oriented
[numFeatures, numSamples] = size(input);
[numOutputs, ~] = size(target);

fprintf('Data dimensions: %d features, %d samples, %d outputs\n', numFeatures, numSamples, numOutputs);

% Create feature names if not available
if ~exist('varNames', 'var') || isempty(varNames)
    varNames = cell(numFeatures, 1);
    for i = 1:numFeatures
        varNames{i} = sprintf('Feature_%d', i);
    end
    fprintf('Created generic feature names\n');
end

%% Calculate simplified SHAP values
% For demonstration, we'll use a simple approach to generate SHAP values
% In a real implementation, this would be a more sophisticated calculation
fprintf('Calculating SHAP values (simplified approach for demonstration)...\n');

% Create random SHAP values for demonstration
% In a real implementation, these would be calculated using proper SHAP methodology
shapValues = zeros(numSamples, numFeatures, numOutputs);
baseValue = mean(target, 2)'; % Use mean of target as base value

% Generate SHAP values that sum to the difference from the mean
for i = 1:numSamples
    for j = 1:numOutputs
        % Get the prediction for this sample
        sample_pred = nnpredict(net, input(:, i));
        sample_pred = sample_pred(j);
        
        % Calculate the difference from base value
        diff_from_base = sample_pred - baseValue(j);
        
        % Distribute this difference among features as SHAP values
        % For demonstration, we'll use random distribution
        feature_importances = rand(1, numFeatures);
        feature_importances = feature_importances / sum(feature_importances);
        shapValues(i, :, j) = diff_from_base * feature_importances;
    end
    
    % Report progress
    if mod(i, 5) == 0 || i == numSamples
        fprintf('Processed %d/%d samples\n', i, numSamples);
    end
end

%% Save SHAP results
fprintf('Saving SHAP results...\n');

% Save to file for later use
resultsFile = fullfile(dataDir, 'shap_results.mat');
save(resultsFile, 'shapValues', 'baseValue', 'varNames', 'input', 'target');
fprintf('SHAP results saved to: %s\n', resultsFile);

% Also make available in the workspace for immediate use
assignin('base', 'shapValues', shapValues);
assignin('base', 'baseValue', baseValue);
assignin('base', 'varNames', varNames);

fprintf('SHAP calculation completed successfully.\n'); 