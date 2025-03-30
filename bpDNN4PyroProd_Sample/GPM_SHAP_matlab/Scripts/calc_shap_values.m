%% SHAP Value Calculation for bpDNN Pyrolysis Model
% This script calculates SHAP values for the trained neural network model
% using the KernelSHAP approach adapted for MATLAB.

% Get variable names for better interpretation
varNames = {'Location', 'VolatileMatters/%', 'FixedCarbon/%', 'Ash/%', ...
            'C/%', 'H/%', 'N/%', 'O/%', 'S/%', 'Temperature/°C', ...
            'ReactionTime/min', 'HeatingRate/(K/min)', 'ReactorType'};

% Add feedstock names
load(fullfile(modelFilesDir, 'CopyrolysisFeedstock.mat'));

% Create a map to track occurrences of each feedstock name
nameCountMap = containers.Map('KeyType', 'char', 'ValueType', 'double');

% Add FeedID_ names with uniqueness check
for i = 1:size(CopyrolysisFeedstockTag, 1)
    feedName = CopyrolysisFeedstockTag{i};
    feedIdName = ['FeedID_' feedName];
    
    if ~nameCountMap.isKey(feedIdName)
        nameCountMap(feedIdName) = 1;
        varNames{end+1} = feedIdName;
    else
        % If name already exists, append a count
        count = nameCountMap(feedIdName);
        varNames{end+1} = [feedIdName '_' num2str(count)];
        nameCountMap(feedIdName) = count + 1;
    end
end

% Add MixRatio_ names with uniqueness check
for i = 1:size(CopyrolysisFeedstockTag, 1)
    feedName = CopyrolysisFeedstockTag{i};
    mixRatioName = ['MixRatio_' feedName];
    
    if ~nameCountMap.isKey(mixRatioName)
        nameCountMap(mixRatioName) = 1;
        varNames{end+1} = mixRatioName;
    else
        % If name already exists, append a count
        count = nameCountMap(mixRatioName);
        varNames{end+1} = [mixRatioName '_' num2str(count)];
        nameCountMap(mixRatioName) = count + 1;
    end
end

% Target names
targetNames = {'Char Yield/%', 'Liquid Yield/%', 'Gas Yield/%'};

%% Prepare data for SHAP analysis
% We'll use all data points for SHAP analysis to get a complete picture
% Ensure we're using data from the trained model in trainingDir
if ~exist('input', 'var') || ~exist('target', 'var')
    % If not already in workspace, load from training directory
    if exist('trainingDir', 'var')
        load(fullfile(trainingDir, 'Results_trained.mat'), 'input', 'target', 'net');
    else
        error('Training data not found. Please run the training script first.');
    end
end

X_all = input';
y_all = target';

% Number of features
numFeatures = size(X_all, 2);
numSamples = size(X_all, 1);
numTargets = size(y_all, 2);

%% Implement KernelSHAP for MATLAB
% This is an implementation of KernelSHAP based on the principles in the SHAP documentation

% Set global variables required for nnpredict
global PS TS;

% Make sure global variables are available
if ~exist('PS', 'var') || isempty(PS) || ~exist('TS', 'var') || isempty(TS)
    % Try to recover them from the net if possible
    if isfield(net, 'PS') && isfield(net, 'TS')
        PS = net.PS;
        TS = net.TS;
    end
end

% Parameters
numBackground = min(100, numSamples); % Background sample size (for baseline)
numCoalitions = 500; % Number of feature coalitions to sample per instance

% Select background samples (random subset for computational efficiency)
bgIdx = randperm(numSamples, numBackground);
X_background = X_all(bgIdx, :);

% Initialize storage for SHAP values
shapValues = zeros(numSamples, numFeatures, numTargets);
baseValue = zeros(1, numTargets);

% Pre-compute log factorials to avoid numerical issues
logfactorials = zeros(numFeatures+1, 1);
for i = 2:numFeatures+1
    logfactorials(i) = logfactorials(i-1) + log(i-1);
end

% Calculate baseline prediction (model expected value) - vectorized approach
bg_predictions = zeros(numBackground, numTargets);
% Process predictions in batches to avoid memory issues
batchSize = 20; % Adjust based on memory constraints
numBatches = ceil(numBackground/batchSize);

for b = 1:numBatches
    startIdx = (b-1)*batchSize + 1;
    endIdx = min(b*batchSize, numBackground);
    
    % Create batch of inputs
    x_batch = X_background(startIdx:endIdx, :)';
    
    % Predict all at once
    pred_batch = nnpredict(net, x_batch);
    
    % Store results
    bg_predictions(startIdx:endIdx, :) = pred_batch';
end

baseValue = mean(bg_predictions, 1);

% Progress tracking
fprintf('Calculating SHAP values for %d samples:\n', numSamples);
progressInterval = max(1, floor(numSamples/20)); % Update every 5%

% Precompute coalition matrices and weights (outside parfor)
coalitionMatrices = cell(numSamples, 1);
weightMatrices = cell(numSamples, 1);

fprintf('Precomputing coalition matrices and weights...\n');
for i = 1:numSamples
    % Pre-allocate for this iteration
    coalitionMatrix = false(numCoalitions, numFeatures);
    weightsMatrix = zeros(numCoalitions, 1);
    
    % Randomly generate coalitions (binary masks of features)
    for c = 1:numCoalitions
        % Random number of features to include
        numIncluded = randi(numFeatures);
        % Randomly select which features to include
        includedFeatures = randperm(numFeatures, numIncluded);
        
        % Create feature coalition
        coalition = false(1, numFeatures);
        coalition(includedFeatures) = true;
        coalitionMatrix(c, :) = coalition;
        
        % Calculate coalition weight (shapley kernel)
        M = numFeatures; % Total number of features
        s = sum(coalition); % Size of the coalition
        
        if s == 0 || s == M
            weight = 1000; % Very large weight for empty and full coalitions
        else
            % Instead of using nchoosek directly which can cause numerical issues:
            % weight = (M - 1) / (nchoosek(M, s) * s * (M - s));
            
            % Use logarithms for numerical stability
            logbinom = logNchoosekStable(M, s, logfactorials);
            logweight = log(M - 1) - (logbinom + log(s) + log(M - s));
            weight = exp(logweight);
            
            % If weight is still problematic, use a simpler approximation
            if isinf(weight) || isnan(weight)
                weight = 1 / (s * (M - s));
            end
        end
        weightsMatrix(c) = weight;
    end
    
    coalitionMatrices{i} = coalitionMatrix;
    weightMatrices{i} = weightsMatrix;
end

% Enable parallel computation with proper setup
useParallel = true;
try
    % Check for MATLAB version and use appropriate parallel pool initialization
    if verLessThan('matlab', '8.5') % MATLAB 2014b is 8.4
        % In MATLAB 2014b, use the older parallel API without matlabpool directly
        try
            % This works for 2014b without using deprecated matlabpool
            if isempty(gcp('nocreate'))
                parpool('local');
                fprintf('Created parallel pool for 2014b\n');
            else
                fprintf('Using existing parallel pool\n');
            end
        catch ME
            useParallel = false;
            warning('ParallelToolbox:Error', '%s', ['Parallel Computing Toolbox not available: ' ME.message]);
        end
    else
        % Newer API for MATLAB 2015a and newer
        poolobj = gcp('nocreate');
        if isempty(poolobj)
            poolobj = parpool('local');
            fprintf('Created parallel pool with %d workers\n', poolobj.NumWorkers);
        else
            fprintf('Using existing parallel pool with %d workers\n', poolobj.NumWorkers);
        end
    end
catch ME
    useParallel = false;
    warning('ParallelToolbox:Error', '%s', ['Parallel Computing Toolbox not available: ' ME.message]);
end

% Store neural network and preprocessing parameters
netStruct = net;
PS_copy = PS;
TS_copy = TS;

% Calculate SHAP values for each instance
if useParallel
    fprintf('Computing SHAP values with parallel computation...\n');
    
    % Make a copy of essential variables for better parfor performance
    X_all_copy = X_all;
    X_background_copy = X_background;
    numCoalitions_copy = numCoalitions;
    numBackground_copy = numBackground;
    numFeatures_copy = numFeatures;
    numTargets_copy = numTargets;
    baseValue_copy = baseValue;
    
    % Initialize results collector for parfor
    shapValues_cell = cell(numSamples, 1);
    
    % Use parfor instead of parfeval for better compatibility with older MATLAB versions
    parfor i = 1:numSamples
        % Get precomputed coalition matrix and weights
        coalitionMatrix_i = coalitionMatrices{i};
        weightsMatrix_i = weightMatrices{i};
        
        % Current instance
        x_instance = X_all_copy(i, :);
        
        % Compute SHAP values for this instance
        instance_shapValues = computeInstanceShap(x_instance, X_background_copy, coalitionMatrix_i, weightsMatrix_i, ...
            netStruct, numCoalitions_copy, numBackground_copy, numFeatures_copy, numTargets_copy, PS_copy, TS_copy);
        
        % Store in a temporary cell array
        shapValues_cell{i} = instance_shapValues;
        
        % No progress display inside parfor (it would be disordered)
    end
    
    % Convert cell results to the final array
    for i = 1:numSamples
        shapValues(i,:,:) = shapValues_cell{i};
        
        % Display progress outside of parfor
        if mod(i, progressInterval) == 0 || i == numSamples
            fprintf('Progress: %.1f%% (%d/%d samples)\n', 100*i/numSamples, i, numSamples);
        end
    end
else
    fprintf('Computing SHAP values with sequential computation...\n');
    % Sequential version (no parfor)
    for i = 1:numSamples
        % Get precomputed coalition matrix and weights
        coalitionMatrix = coalitionMatrices{i};
        weightsMatrix = weightMatrices{i};
        
        % Current instance
        x_instance = X_all(i, :);
        
        % Compute SHAP values for this instance
        instance_shapValues = computeInstanceShap(x_instance, X_background, coalitionMatrix, weightsMatrix, ...
            netStruct, numCoalitions, numBackground, numFeatures, numTargets, PS_copy, TS_copy);
        
        % Store results
        shapValues(i,:,:) = instance_shapValues;
        
        % Display progress
        if mod(i, progressInterval) == 0 || i == numSamples
            fprintf('Progress: %.1f%% (%d/%d samples)\n', 100*i/numSamples, i, numSamples);
        end
    end
end

%% Save SHAP values and related data
% Create a subdirectory for data files
dataDir = fullfile(shapDir, 'Data');
if ~exist(dataDir, 'dir')
    mkdir(dataDir);
    disp(['Created data directory: ' dataDir]);
else
    disp(['Using existing data directory: ' dataDir]);
end

% Create the variables that plot_shap_results.m and plot_shap_beeswarm.m expect
featureNames = varNames;
input = X_all';  % Transpose back to match the original input format

% Ensure global variables are saved with the results
if exist('PS', 'var') && exist('TS', 'var')
    PS_global = PS;
    TS_global = TS;
    fprintf('Adding global preprocessing variables PS_global and TS_global to results\n');
end

% Try to calculate model predictions
disp('Calculating model predictions for force plots...');
try
    if exist('net', 'var') && exist('input', 'var')
        try
            % First try using nnpredict which is safer
            predictions = nnpredict(net, input);
            fprintf('Predictions calculated successfully for %d samples using nnpredict\n', size(input, 2));
        catch predErr
            % Fall back to direct network call with input validation
            if ~isempty(input) && isnumeric(input) && all(size(input) > 0)
                predictions = net(input);
                fprintf('Predictions calculated successfully for %d samples using direct network call\n', size(input, 2));
            else
                warning('Prediction:InvalidInput', '%s', ['Invalid input data for prediction: ' predErr.message]);
                % Initialize empty predictions to avoid warnings in later code
                predictions = [];
            end
        end
    else
        warning('Prediction:MissingVars', 'Failed to calculate predictions: Required variables not found');
        % Initialize empty predictions to avoid warnings in later code
        predictions = [];
    end
catch ME
    warning('Prediction:Error', '%s', ['Failed to calculate predictions: ' ME.message]);
    % Initialize empty predictions to avoid warnings in later code
    predictions = [];
end

% Save results and preprocessing variables
try
    if exist('PS_global', 'var') && exist('TS_global', 'var')
        fprintf('Adding global preprocessing variables PS_global and TS_global to results\n');
        if exist('predictions', 'var') && ~isempty(predictions)
            save(fullfile(dataDir, 'shap_results.mat'), 'shapValues', 'baseValue', 'varNames', ...
                'input', 'target', 'X_background', 'predictions', 'PS_global', 'TS_global');
            fprintf('Saved SHAP results with predictions and global variables\n');
        else
            save(fullfile(dataDir, 'shap_results.mat'), 'shapValues', 'baseValue', 'varNames', ...
                'input', 'target', 'X_background', 'PS_global', 'TS_global');
            fprintf('Saved SHAP results with global variables (no predictions)\n');
        end
    else
        if exist('predictions', 'var') && ~isempty(predictions)
            save(fullfile(dataDir, 'shap_results.mat'), 'shapValues', 'baseValue', 'varNames', ...
                'input', 'target', 'X_background', 'predictions');
            fprintf('Saved SHAP results with predictions (no global variables)\n');
        else
            save(fullfile(dataDir, 'shap_results.mat'), 'shapValues', 'baseValue', 'varNames', ...
                'input', 'target', 'X_background');
            fprintf('Saved SHAP results (no predictions or global variables)\n');
        end
    end
catch ME
    warning('Save:Error', '%s', ['Error saving SHAP results: ' ME.message]);
end

% Helper function to compute log of binomial coefficient
function logbinom = logNchoosekStable(n, k, logfactorial_array)
    % Using the formula log(n choose k) = log(n!) - log(k!) - log((n-k)!)
    % Pre-computed log factorials are used
    if k == 0 || k == n
        logbinom = 0; % log(1) = 0, since nCr(n,0) = nCr(n,n) = 1
    else
        logbinom = logfactorial_array(n+1) - logfactorial_array(k+1) - logfactorial_array(n-k+1);
    end
end

% Helper function to compute SHAP values for a single instance
function instance_shapValues = computeInstanceShap(x_instance, X_background, coalitionMatrix, weightsMatrix, ...
    net, numCoalitions, numBackground, numFeatures, numTargets, PS_local, TS_local)
    % Set global variables for this worker
    global PS TS;
    PS = PS_local;
    TS = TS_local;
    
    % Get predictions for each coalition (vectorized approach)
    predictions = zeros(numCoalitions, numTargets);
    
    % Process in batches to optimize memory usage
    batchSize = 10; % Adjust based on memory constraints
    numBatches = ceil(numCoalitions/batchSize);
    
    for batch = 1:numBatches
        startIdx = (batch-1)*batchSize + 1;
        endIdx = min(batch*batchSize, numCoalitions);
        batchCount = endIdx - startIdx + 1;
        
        % Pre-allocate batch results
        batch_predictions = zeros(batchCount, numTargets);
        
        % Process each coalition in the batch
        for c_idx = 1:batchCount
            c = startIdx + c_idx - 1;
            
            % Create masked instances efficiently
            mask = ~coalitionMatrix(c, :);
            
            % Matrix approach: replicate x_instance and replace masked features with background values
            masked_instances = repmat(x_instance, numBackground, 1);
            masked_instances(:, mask) = X_background(:, mask);
            
            % Predict for all masked instances at once
            pred_results = zeros(numBackground, numTargets);
            
            % Batch predictions for this coalition
            pred_all = nnpredict(net, masked_instances');
            pred_results = pred_all';
            
            % Average predictions
            batch_predictions(c_idx, :) = mean(pred_results, 1);
        end
        
        % Store batch results
        predictions(startIdx:endIdx, :) = batch_predictions;
    end
    
    % Solve weighted least squares for each target
    instance_shapValues = zeros(numFeatures, numTargets);
    
    % Precompute common terms for all targets
    X_coalition = double(coalitionMatrix);
    X_coalition_with_intercept = [ones(numCoalitions, 1), X_coalition];
    W = diag(weightsMatrix);
    
    % Compute (X'WX + λI)^-1 X'W once for efficiency
    lambda = 1e-8; % Small regularization parameter
    I = eye(size(X_coalition_with_intercept, 2));
    XtWX_lambda = X_coalition_with_intercept' * W * X_coalition_with_intercept + lambda * I;
    XtW = X_coalition_with_intercept' * W;
    
    % Use matrix operations for all targets at once if possible
    try
        % Try to solve for all targets at once (more efficient)
        coeffs = XtWX_lambda \ (XtW * predictions);
        
        % First row contains intercepts (base values)
        % Remaining rows are the SHAP values
        instance_shapValues = coeffs(2:end, :);
    catch
        % Fall back to solving for each target separately
        for t = 1:numTargets
            y_pred = predictions(:, t);
            coeffs = XtWX_lambda \ (XtW * y_pred);
            instance_shapValues(:, t) = coeffs(2:end);
        end
    end
end 