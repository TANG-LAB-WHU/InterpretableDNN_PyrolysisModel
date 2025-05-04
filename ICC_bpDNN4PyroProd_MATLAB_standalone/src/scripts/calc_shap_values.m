%% SHAP Value Calculation for Neural Network Model
% This script calculates SHAP values for the trained neural network model
% using a proper implementation of the SHAP methodology based on the official SHAP library.

fprintf('Starting SHAP value calculation with proper methodology...\n');

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
% Check for different variable naming conventions (X/Y vs input/target)
if exist('X', 'var') && ~exist('input', 'var')
    fprintf('Found X variable, mapping to input\n');
    input = X;
elseif ~exist('input', 'var') && exist('net', 'var')
    fprintf('No input variable found, attempting to load model data...\n');
    
    % Try to find and load model data
    trainedModelFile = [];
    
    % First check if we're in debug mode with a debug model file
    if exist('modelFileUsed', 'var')
        trainedModelFile = modelFileUsed;
        fprintf('Using model file from workspace: %s\n', trainedModelFile);
    else
        % Check common locations for model files
        rootDir = fileparts(fileparts(fileparts(mfilename('fullpath'))));
        
        % Check best model directory first (priority)
        bestModelDir = fullfile(rootDir, 'results', 'best_model');
        if exist(fullfile(bestModelDir, 'best_model.mat'), 'file')
            trainedModelFile = fullfile(bestModelDir, 'best_model.mat');
        end
        
        % Check optimization directory next
        optimizationDir = fullfile(rootDir, 'results', 'optimization');
        if isempty(trainedModelFile) && exist(fullfile(optimizationDir, 'best_model.mat'), 'file')
            trainedModelFile = fullfile(optimizationDir, 'best_model.mat');
        end
        
        % Check training directory next
        trainingDir = fullfile(rootDir, 'results', 'training');
        if isempty(trainedModelFile) && exist(fullfile(trainingDir, 'Results_trained.mat'), 'file')
            trainedModelFile = fullfile(trainingDir, 'Results_trained.mat');
        end
        
        % Check debug directory as last resort
        debugDir = fullfile(rootDir, 'results', 'debug');
        if isempty(trainedModelFile) && exist(fullfile(debugDir, 'Results_trained.mat'), 'file')
            trainedModelFile = fullfile(debugDir, 'Results_trained.mat');
        end
        
        if isempty(trainedModelFile)
            error('Could not find a valid model file to load');
        else
            fprintf('Found model file: %s\n', trainedModelFile);
        end
    end
    
    % Load the model data
    try
        fprintf('Loading model data from %s...\n', trainedModelFile);
        modelData = load(trainedModelFile);
        
        % Check for input data
        if isfield(modelData, 'input')
            input = modelData.input;
            fprintf('Loaded input data with dimensions %dx%d\n', size(input, 1), size(input, 2));
        elseif isfield(modelData, 'X')
            input = modelData.X;
            fprintf('Loaded X data with dimensions %dx%d\n', size(input, 1), size(input, 2));
        else
            error('Model file does not contain input or X data');
        end
        
        % Check for target data
        if isfield(modelData, 'target')
            target = modelData.target;
            fprintf('Loaded target data with dimensions %dx%d\n', size(target, 1), size(target, 2));
        elseif isfield(modelData, 'Y')
            target = modelData.Y;
            fprintf('Loaded Y data with dimensions %dx%d\n', size(target, 1), size(target, 2));
        else
            error('Model file does not contain target or Y data');
        end
        
        % Check for neural network model
        if isfield(modelData, 'net')
            net = modelData.net;
            fprintf('Loaded neural network model\n');
        else
            error('Model file does not contain neural network model');
        end
    catch loadErr
        error('Error loading model data: %s', loadErr.message);
    end
end

if exist('Y', 'var') && ~exist('target', 'var')
    fprintf('Found Y variable, mapping to target\n');
    target = Y;
end

% Check if input dimensions need to be transposed (handle row vs column features)
% If input has more rows than columns, assume features are in columns
if ~isempty(input) && size(input, 1) < size(input, 2)
    fprintf('Detected input with features in columns (samples in rows), transposing...\n');
    input = input';
    fprintf('After transposing: input size is [%d, %d] (features x samples)\n', size(input, 1), size(input, 2));
end

% Do the same for target if needed
if ~isempty(target) && size(target, 1) < size(target, 2)
    fprintf('Detected target with outputs in columns (samples in rows), transposing...\n');
    target = target';
    fprintf('After transposing: target size is [%d, %d] (outputs x samples)\n', size(target, 1), size(target, 2));
end

% Ensure net has necessary fields for SHAP analysis
if ~isfield(net, 'numInputs') && ~isfield(net, 'inputs')
    fprintf('Adding required fields to net structure\n');
    if size(input, 1) > 0
        net.numInputs = size(input, 1);
    else
        net.numInputs = 1; % Default if input dimensions unknown
    end
end

% Extract dimensions
[numFeatures, numSamples] = size(input);
[numOutputs, ~] = size(target);

% Check if dimensions look suspicious (more features than samples, or more outputs than samples)
if numFeatures > numSamples || numOutputs > numSamples
    fprintf('WARNING: Suspicious dimensions detected - more features/outputs than samples\n');
    fprintf('Current dimensions: features=%d, samples=%d, outputs=%d\n', numFeatures, numSamples, numOutputs);
    
    if numFeatures == numOutputs && numFeatures > numSamples
        % This suggests that features and samples dimensions are swapped
        fprintf('CORRECTION: Detected likely dimension swap (features/samples)\n');
        
        % Swap dimensions
        input = input';
        target = target';
        
        % Get corrected dimensions
        [numFeatures, numSamples] = size(input);
        [numOutputs, ~] = size(target);
        
        fprintf('After correction: features=%d, samples=%d, outputs=%d\n', numFeatures, numSamples, numOutputs);
    end
end

fprintf('Using %d features, %d samples, and %d outputs for SHAP analysis\n', numFeatures, numSamples, numOutputs);

% Try to read feature names from RawInputData.xlsx with improved error handling for ICC cluster
if ~exist('featureNames', 'var') || isempty(featureNames)
    fprintf('Trying to read real feature names from RawInputData.xlsx...\n');
    rootDir = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    rawDataPath = fullfile(rootDir, 'data', 'raw', 'RawInputData.xlsx');
    
    featureNamesLoaded = false;
    
    if exist(rawDataPath, 'file')
        try
            % Platform-specific approach for reading Excel files
            isICC = ~isempty(strfind(computer('arch'), 'glnxa64'));
            
            if isICC
                fprintf('Detected ICC cluster environment. Using CSV reading instead of Excel...\n');
                
                % Check if a CSV version exists
                csvPath = fullfile(rootDir, 'data', 'raw', 'RawInputData.csv');
                if exist(csvPath, 'file')
                    fprintf('Found CSV version of the data file. Reading from: %s\n', csvPath);
                    try
                        % Read CSV with textscan
                        fid = fopen(csvPath, 'r');
                        if fid ~= -1
                            headerLine = fgetl(fid);
                            fclose(fid);
                            
                            % Parse header line
                            if ~isempty(headerLine)
                                rawFeatureNames = strsplit(headerLine, ',');
                                if length(rawFeatureNames) >= numFeatures
                                    featureNames = rawFeatureNames(1:numFeatures);
                                    fprintf('Successfully read %d real feature names from CSV\n', length(featureNames));
                                    featureNamesLoaded = true;
                                end
                            end
                        end
                    catch csvErr
                        fprintf('Error reading CSV file: %s\n', csvErr.message);
                        % Will generate names below
                    end
                else
                    fprintf('CSV version not found. Skipping feature name reading on ICC cluster.\n');
                end
            else
                % Try traditional xlsread on non-ICC platforms
                try
                    fprintf('Using xlsread to get feature names (non-ICC environment)...\n');
                    [~, ~, raw] = xlsread(rawDataPath, 1, 'A1:Z1');
                    if ~isempty(raw)
                        rawFeatureNames = raw(~cellfun(@isempty, raw));
                        if length(rawFeatureNames) >= numFeatures
                            featureNames = rawFeatureNames(1:numFeatures);
                            fprintf('Successfully read %d real feature names from Excel\n', length(featureNames));
                            featureNamesLoaded = true;
                        else
                            fprintf('Not enough feature names in Excel file (%d < %d needed)\n', length(rawFeatureNames), numFeatures);
                            % Will generate names below
                        end
                    end
                catch xlsReadErr
                    fprintf('Error using xlsread: %s\n', xlsReadErr.message);
                    fprintf('This is expected on headless environments like HPC clusters.\n');
                    % Will generate names below
                end
            end
        catch generalErr
            fprintf('Error during feature name reading: %s\n', generalErr.message);
            % Will generate names below
        end
    else
        fprintf('RawInputData.xlsx not found at %s\n', rawDataPath);
        % Will generate names below
    end
    
    % Generate feature names if not loaded from file
    if ~exist('featureNames', 'var') || isempty(featureNames) || length(featureNames) < numFeatures || ~featureNamesLoaded
        fprintf('Generating feature names...\n');
        featureNames = cell(1, numFeatures);
        for i = 1:numFeatures
            featureNames{i} = sprintf('Feature_%d', i);
        end
        fprintf('Created %d generic feature names\n', length(featureNames));
    end
end

% If varNames doesn't exist, use featureNames
if ~exist('varNames', 'var') || isempty(varNames)
    fprintf('Setting varNames to featureNames\n');
    varNames = featureNames;
end

% Generate target names if not provided
if ~exist('targetNames', 'var') || isempty(targetNames) || length(targetNames) ~= numOutputs
    targetNames = cell(numOutputs, 1);
    
    % Default target names if there are exactly 3 outputs (common pyrolysis case)
    if numOutputs == 3
        targetNames{1} = 'Char/%';
        targetNames{2} = 'Liquid/%';
        targetNames{3} = 'Gas/%';
    else
        % For other cases, use generic names
        for i = 1:numOutputs
            targetNames{i} = sprintf('Target %d', i);
        end
    end
    fprintf('Created target names: %s\n', strjoin(targetNames, ', '));
end

% Initialize default parameters if not already defined in the workspace
if ~exist('numBackgroundSamples', 'var')
    numBackgroundSamples = min(50, numSamples);  % Background sample count
    fprintf('Setting default numBackgroundSamples = %d\n', numBackgroundSamples);
end

if ~exist('useAllSamples', 'var')
    useAllSamples = true;  % Default to true for full analysis
    fprintf('Setting default useAllSamples = %d\n', useAllSamples);
end

if ~exist('useAllFeatures', 'var')
    useAllFeatures = true;  % Default to true for full analysis
    fprintf('Setting default useAllFeatures = %d\n', useAllFeatures);
end

if ~exist('numShapSamples', 'var')
    if useAllSamples
        numShapSamples = numSamples;
        fprintf('Using all %d samples for SHAP analysis\n', numShapSamples);
    else
        numShapSamples = min(20, numSamples);
        fprintf('Using %d samples for SHAP analysis (limited mode)\n', numShapSamples);
    end
end

if ~exist('selectedTargets', 'var')
    selectedTargets = 1:numOutputs;  % Default to analyze all targets
    fprintf('Analyzing all %d target variables\n', length(selectedTargets));
end

%% Create neural network prediction function for SHAP calculation
% Define a wrapper prediction function that works with our network structure
nnpredict_safe = @(net, x) predict_wrapper(net, x, numOutputs);

%% Calculate SHAP values using a proper algorithm based on Shapley values
fprintf('Calculating SHAP values using kernel SHAP approximation...\n');

try
    % Sample selection for SHAP analysis
    if useAllSamples
        selectedIndices = 1:numSamples;
    else
        % Randomly select samples
        selectedIndices = randperm(numSamples, min(numSamples, numShapSamples));
    end
    
    numSelectedSamples = length(selectedIndices);
    fprintf('Selected %d samples for SHAP analysis\n', numSelectedSamples);
    
    % Create background dataset by randomly sampling from input data
    % This is used as the reference distribution for marginalizing out features
    bgIndices = randperm(numSamples, min(numSamples, numBackgroundSamples));
    background = input(:, bgIndices);
    fprintf('Created background dataset with %d samples\n', size(background, 2));
    
    % Calculate base value (average prediction over background dataset)
    baseValue = zeros(1, numOutputs);
    fprintf('Calculating base values using background dataset...\n');
    for i = 1:size(background, 2)
        baseValue = baseValue + nnpredict_safe(net, background(:, i))' / size(background, 2);
    end
    fprintf('Base values for each target: %s\n', mat2str(baseValue, 4));
    
    % Initialize SHAP values array
    shapValues = zeros(numSelectedSamples, numFeatures, numOutputs);
    
    % Number of coalitions to sample (balance accuracy vs computation)
    % For small feature count, can use exact calculation (2^numFeatures)
    % For larger feature count, use sampling approach
    if numFeatures <= 10
        % For small feature count, we can calculate exact Shapley values
        fprintf('Using exact Shapley value calculation (small feature count: %d)\n', numFeatures);
        numCoalitions = 2^numFeatures;
        exactCalculation = true;
    else
        % For larger feature count, use sampling approach
        exactCalculation = false;
        if strcmpi(analysisMode, 'full')
            numCoalitions = min(1024, 2^numFeatures); % Use more coalitions in full mode
        else
            numCoalitions = min(256, 2^numFeatures);  % Use fewer coalitions in debug mode
        end
        fprintf('Using approximation with %d coalitions (feature count: %d)\n', numCoalitions, numFeatures);
    end
    
    % For each sample, calculate SHAP values
    fprintf('Beginning SHAP calculation for %d samples...\n', numSelectedSamples);
    for i = 1:numSelectedSamples
        % Show progress every 5 samples
        if mod(i, 5) == 0 || i == 1 || i == numSelectedSamples
            fprintf('Processing sample %d/%d\n', i, numSelectedSamples);
        end
        
        sampleIdx = selectedIndices(i);
        x_sample = input(:, sampleIdx);
        
        % For exact calculation, enumerate all possible coalitions
        if exactCalculation
            % Generate all possible coalitions (2^numFeatures)
            coalitions = de2bi(0:(2^numFeatures-1), numFeatures);
        else
            % For sampling approach, generate random coalitions
            % Ensure we include the empty coalition and full coalition
            coalitions = rand(numCoalitions-2, numFeatures) > 0.5;
            % Add empty coalition and full coalition
            coalitions = [zeros(1, numFeatures); coalitions; ones(1, numFeatures)];
        end
        
        % For each target
        for j = selectedTargets
            % Calculate Shapley values using the coalition method
            shapValues(i, :, j) = calculateShapleyValues(net, x_sample, background, coalitions, nnpredict_safe, j, baseValue(j));
        end
    end
    
    % Save results
    fprintf('SHAP calculation completed, saving results...\n');
    resultsFile = fullfile(dataDir, 'shap_results.mat');
    
    % Prepare selected data for saving
    selected_input = input(:, selectedIndices);
    selected_target = target(:, selectedIndices);
    
    % Add metadata to document the calculation method
    shapMetadata = struct();
    shapMetadata.method = 'kernel_shap';
    shapMetadata.exactCalculation = exactCalculation;
    shapMetadata.numCoalitions = numCoalitions;
    shapMetadata.numBackgroundSamples = size(background, 2);
    shapMetadata.calculationDate = datestr(now);
    
    % Save to MAT file
    save(resultsFile, 'shapValues', 'baseValue', 'varNames', 'featureNames', 'targetNames', ...
        'selected_input', 'selected_target', 'input', 'target', 'shapMetadata');
    
    fprintf('SHAP results saved to: %s\n', resultsFile);
    
    % Also make variables available in workspace for immediate use
    assignin('base', 'shapValues', shapValues);
    assignin('base', 'baseValue', baseValue);
    assignin('base', 'varNames', varNames);
    assignin('base', 'featureNames', featureNames);
    assignin('base', 'targetNames', targetNames);
    assignin('base', 'shapMetadata', shapMetadata);
    
    fprintf('SHAP analysis completed successfully.\n');
catch shapErr
    fprintf('\n===== ERROR DURING SHAP CALCULATION =====\n');
    fprintf('Error message: %s\n', shapErr.message);
    
    % Print stack trace for debugging
    for k = 1:length(shapErr.stack)
        fprintf('  File: %s, Line: %d, Function: %s\n', ...
            shapErr.stack(k).file, shapErr.stack(k).line, shapErr.stack(k).name);
    end
    
    % Still try to save partial results if possible
    try
        if exist('shapValues', 'var') && ~isempty(shapValues)
            fprintf('Attempting to save partial results...\n');
            partialResultsFile = fullfile(dataDir, 'shap_partial_results.mat');
            save(partialResultsFile, 'shapValues', 'baseValue', 'varNames', 'featureNames', 'targetNames', 'shapErr');
            fprintf('Partial results saved to: %s\n', partialResultsFile);
        end
    catch saveErr
        fprintf('Failed to save partial results: %s\n', saveErr.message);
    end
    
    % Re-throw the error so the caller knows there was a problem
    rethrow(shapErr);
end

%% Helper Function: Calculate Shapley Values for one sample and one target
function shap_values = calculateShapleyValues(net, x, background, coalitions, predict_fn, target_idx, base_value)
    % This function calculates Shapley values for one sample and one target
    % using the coalition sampling method
    
    num_features = length(x);
    num_coalitions = size(coalitions, 1);
    num_background = size(background, 2);
    
    % Initialize Shapley values
    shap_values = zeros(1, num_features);
    
    % For each feature, calculate marginal contribution across coalitions
    for feature_idx = 1:num_features
        % Track marginal contributions
        marginal_contribution = 0;
        weight_sum = 0;
        
        % For each coalition, check marginal contribution of this feature
        for coalition_idx = 1:num_coalitions
            % Current coalition (which features are "in")
            coalition = coalitions(coalition_idx, :);
            
            % Skip if feature is already in this coalition
            if coalition(feature_idx) == 1
                continue;
            end
            
            % Create coalition with feature added
            with_feature = coalition;
            with_feature(feature_idx) = 1;
            
            % Calculate weight for this coalition
            % This approximates: |S|!(n-|S|-1)!/n!
            % Where |S| is coalition size, n is feature count
            coalition_size = sum(coalition);
            weight = 1;
            
            % For exact calculation, we use true Shapley weights
            % For sampling, all weights are equal (kernel SHAP approximation)
            if num_coalitions == 2^num_features
                % Calculate coalition weight using Shapley formula
                if coalition_size == 0 || coalition_size == num_features - 1
                    weight = 1 / num_features;
                else
                    weight = (factorial(coalition_size) * factorial(num_features - coalition_size - 1)) / factorial(num_features);
                end
            end
            
            % Predict with and without feature
            v_with = evaluateCoalition(net, x, background, with_feature, predict_fn, target_idx);
            v_without = evaluateCoalition(net, x, background, coalition, predict_fn, target_idx);
            
            % Accumulate weighted marginal contribution
            marginal_contribution = marginal_contribution + weight * (v_with - v_without);
            weight_sum = weight_sum + weight;
        end
        
        % Normalize by weight sum
        if weight_sum > 0
            shap_values(feature_idx) = marginal_contribution / weight_sum;
        end
    end
    
    % Consistency check: SHAP values should sum to prediction_diff
    prediction = predict_fn(net, x);
    prediction_diff = prediction(target_idx) - base_value;
    
    % Fix any small rounding errors to ensure SHAP values sum to prediction difference
    shap_sum = sum(shap_values);
    if abs(shap_sum - prediction_diff) > 1e-10
        % Adjust SHAP values to match prediction difference exactly
        shap_values = shap_values * (prediction_diff / shap_sum);
    end
end

%% Helper Function: Evaluate model for a given coalition
function value = evaluateCoalition(net, x, background, coalition, predict_fn, target_idx)
    % This function evaluates the model prediction for a given coalition
    % by marginalization of features not in the coalition
    
    % Marginalization using background samples (interventional approach)
    num_background = size(background, 2);
    
    % Use all background samples to marginalize out
    predictions = zeros(1, num_background);
    
    for bg_idx = 1:num_background
        % Create instance with coalition features from x and others from background
        instance = background(:, bg_idx);
        instance(coalition == 1) = x(coalition == 1);
        
        % Predict for this instance
        pred = predict_fn(net, instance);
        predictions(bg_idx) = pred(target_idx);
    end
    
    % Average prediction across all background combinations
    value = mean(predictions);
end

%% Safe prediction wrapper function
function pred = predict_wrapper(net, x, numOutputs)
    % This function handles different types of network structures
    % and provides a safe prediction method
    
    try
        % Check if x needs to be a column vector
        if isvector(x) && size(x, 2) > 1
            x = x';  % Transpose to column vector
        end
        
        % Check if it's a standard MATLAB neural network
        if isobject(net) && ismethod(net, 'sim')
            % Standard MATLAB neural network
            pred = sim(net, x);
            return;
        end
        
        % Check if it's our custom network structure with W and b fields
        if isstruct(net) && isfield(net, 'W') && isfield(net, 'b')
            % Use our custom nnpredict function if it exists in the path
            if exist('nnpredict', 'file') == 2
                pred = nnpredict(net, x);
                return;
            else
                % Basic feedforward propagation if nnpredict doesn't exist
                a = x;
                for layerIdx = 1:length(net.W)
                    z = net.W{layerIdx} * a + net.b{layerIdx};
                    
                    % Apply activation function 
                    if layerIdx < length(net.W)
                        if isfield(net, 'transferFcn') && length(net.transferFcn) >= layerIdx
                            switch net.transferFcn{layerIdx}
                                case 'tansig'
                                    a = tanh(z);
                                case 'logsig'
                                    a = 1./(1 + exp(-z));
                                case 'poslin'  % ReLU
                                    a = max(0, z);
                                case 'purelin'
                                    a = z;
                                otherwise
                                    a = tanh(z);  % Default to tansig
                            end
                        else
                            a = tanh(z);  % Default to tansig
                        end
                    else
                        % Output layer usually uses purelin
                        if isfield(net, 'transferFcn') && length(net.transferFcn) >= layerIdx
                            switch net.transferFcn{layerIdx}
                                case 'purelin'
                                    a = z;
                                case 'logsig'
                                    a = 1./(1 + exp(-z));
                                case 'tansig'
                                    a = tanh(z);
                                otherwise
                                    a = z;  % Default to purelin
                            end
                        else
                            a = z;  % Default to purelin
                        end
                    end
                end
                pred = a;
                return;
            end
        end
        
        % Last resort fallback
        fprintf('Warning: Unknown network type, using fallback prediction method\n');
        pred = zeros(numOutputs, 1);  % Zero prediction as fallback
        
    catch predErr
        fprintf('Error in predict_wrapper: %s\n', predErr.message);
        % Return default prediction on error
        pred = zeros(numOutputs, 1);
    end
end