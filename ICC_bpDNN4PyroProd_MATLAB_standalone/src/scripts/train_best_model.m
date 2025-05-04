%% Train Best Model Script
% This script loads the best hyperparameters from optimization results
% and trains the final neural network model using the full dataset
% according to the specified strategy (TVT or TT).

% Start timer
startTime = tic;

% Initialize error variable for final check
e = []; % Initialize empty error object

% Setup diary for logging
% Construct a unique log file name if desired, or keep it fixed
logFileName = fullfile(fileparts(mfilename('fullpath')), 'train_best_model_log.txt');
if exist(logFileName, 'file')
    delete(logFileName); % Delete old log file
end
diary(logFileName);
diary on;

fprintf('===== BEST MODEL TRAINING STARTED =====\n');
fprintf('Starting train_best_model.m\n');
fprintf('Script execution started at: %s\n', datestr(now));
fprintf('======================================\n\n');

try % Main try block for the whole script
    %% Setup directories and paths
    fprintf('--- Setting up directories and paths ---\n');
    % Get the root directory based on the script's location
    scriptPath = mfilename('fullpath');
    scriptDir = fileparts(scriptPath);
    rootDir = fileparts(fileparts(scriptDir)); % Assumes script is in src/scripts
    fprintf('Root directory determined: %s\n', rootDir);

    % Define standard paths
    modelDir = fullfile(rootDir, 'src', 'model');
    resultsDir = fullfile(rootDir, 'results');
    bestModelDir = fullfile(resultsDir, 'best_model');
    trainingDir = fullfile(resultsDir, 'training'); % Used for saving a copy
    optimizationDir = fullfile(resultsDir, 'optimization'); % Source of best_config
    dataDir = fullfile(rootDir, 'data', 'processed');
    dataRawDir = fullfile(rootDir, 'data', 'raw');

    % Create results directories if they don't exist
    if ~exist(resultsDir, 'dir'), mkdir(resultsDir); fprintf('Created directory: %s\n', resultsDir); end
    if ~exist(bestModelDir, 'dir'), mkdir(bestModelDir); fprintf('Created directory: %s\n', bestModelDir); end
    if ~exist(trainingDir, 'dir'), mkdir(trainingDir); fprintf('Created directory: %s\n', trainingDir); end

    % Figures directory (within bestModelDir)
    figuresDir = fullfile(bestModelDir, 'Figures');
    if ~exist(figuresDir, 'dir'), mkdir(figuresDir); fprintf('Created directory: %s\n', figuresDir); end

    % Add model directory to path (contains custom nn functions)
    addpath(modelDir);
    fprintf('Added model directory to path: %s\n', modelDir);

    % Check for required custom model functions
    requiredFunctions = {'nncreate', 'nnpredict', 'nnpreprocess', 'nntrain', 'nneval', 'nnpostprocess'}; % Add others if needed
    fprintf('Checking for required model functions in %s...\n', modelDir);
    for i = 1:length(requiredFunctions)
        if ~exist(fullfile(modelDir, [requiredFunctions{i} '.m']), 'file')
            error('Required function not found: %s. Ensure it is in the model directory.', fullfile(modelDir, [requiredFunctions{i} '.m']));
        end
    end
    fprintf('All required model functions found.\n');

    %% Check Data Files
    fprintf('\n--- Checking for required data files ---\n');
    feedstockFile = fullfile(dataDir, 'CopyrolysisFeedstock.mat');
    rawDataFile = fullfile(dataRawDir, 'RawInputData.xlsx');

    % Check primary locations
    feedstockExists = exist(feedstockFile, 'file');
    rawDataExists = exist(rawDataFile, 'file');

    fprintf('Checking primary data locations:\n');
    fprintf('  Feedstock file (%s): %s\n', feedstockFile, iif(feedstockExists, 'Exists', 'Not found'));
    fprintf('  Raw data file (%s): %s\n', rawDataFile, iif(rawDataExists, 'Exists', 'Not found'));

    % Fallback: Check model directory (less ideal)
    if ~feedstockExists
        modelFeedstockFile = fullfile(modelDir, 'CopyrolysisFeedstock.mat');
        if exist(modelFeedstockFile, 'file')
            feedstockFile = modelFeedstockFile;
            feedstockExists = true;
            fprintf('INFO: Using feedstock file found in model directory: %s\n', feedstockFile);
        end
    end
    if ~rawDataExists
        modelRawDataFile = fullfile(modelDir, 'RawInputData.xlsx');
        if exist(modelRawDataFile, 'file')
            rawDataFile = modelRawDataFile;
            rawDataExists = true;
            fprintf('INFO: Using raw data file found in model directory: %s\n', rawDataFile);
        end
    end

    % Ensure data files were found somewhere
    if ~feedstockExists || ~rawDataExists
        error('Required data file(s) not found. Checked primary locations (data/processed, data/raw) and fallback (%s).', modelDir);
    end
    fprintf('Data files located successfully.\n');

    %% Load Best Hyperparameters from Optimization Results
    fprintf('\n--- Loading best hyperparameters ---\n');
    optimResultsFile = fullfile(optimizationDir, 'best_model.mat'); % Standard filename from optimize script
    if ~exist(optimResultsFile, 'file')
        error('ERROR: Best model optimization results not found at %s. Run optimization workflow step first.', optimResultsFile);
    end

    fprintf('Loading optimization results from: %s\n', optimResultsFile);
    optimResults = load(optimResultsFile);

    % Check for the structure containing best parameters
    if isfield(optimResults, 'best_config')
        best_config = optimResults.best_config;
        fprintf('Best configuration loaded from ''best_config'' field.\n');
    elseif isfield(optimResults, 'bestParams') % Handle alternative naming
        best_config = optimResults.bestParams;
        fprintf('Best configuration loaded from ''bestParams'' field.\n');
    else
        error('ERROR: Invalid optimization results file (%s). Missing ''best_config'' or ''bestParams'' field.', optimResultsFile);
    end

    if ~isstruct(best_config)
         error('ERROR: Loaded ''best_config'' or ''bestParams'' is not a structure.');
    end
    fprintf('Best configuration details loaded:\n');
    disp(best_config); % Display the loaded structure

    %% Determine Data Dimensions and Model Parameters
    fprintf('\n--- Determining data dimensions and model parameters ---\n');
    try
        % --- Explicitly define expected data structure ---
        expectedHeaderRow = 1;
        expectedInputCols = 1:13; 
        expectedTargetCols = 14:16; 
        fprintf('Assuming data structure:\n Header Row=%d, Input Cols=%s, Target Cols=%s\n', ...
                expectedHeaderRow, mat2str(expectedInputCols), mat2str(expectedTargetCols));
        % --- End explicit definition ---

        % Read RawInputData.xlsx to get sample count and verify structure
        [~, ~, rawExcelData] = xlsread(rawDataFile);
        if isempty(rawExcelData)
            error('RawInputData.xlsx (%s) appears to be empty.', rawDataFile);
        end
        [numRows, numCols] = size(rawExcelData);

        if numRows <= expectedHeaderRow
            error('RawInputData.xlsx has too few rows (%d) to contain header (row %d) and data.', numRows, expectedHeaderRow);
        end
        headerRow = expectedHeaderRow;
        numSamples = numRows - headerRow; % Number of data samples
        fprintf('RawInputData.xlsx: Found %d samples (Rows %d to %d).\n', numSamples, headerRow + 1, numRows);

        % Validate column ranges against actual columns
        inputFeatureEndCol = max(expectedInputCols);
        targetStartCol = min(expectedTargetCols);
        targetEndCol = max(expectedTargetCols);
        numTargetCols = length(expectedTargetCols);
        if inputFeatureEndCol > numCols || targetStartCol > numCols || targetEndCol > numCols
            error('Defined column ranges (Inputs: %d, Targets: %d-%d) exceed actual columns (%d) in RawInputData.xlsx.', ...
                  inputFeatureEndCol, targetStartCol, targetEndCol, numCols);
        end

        % Determine FeedType_size and numFeatures from CopyrolysisFeedstock.mat
        fprintf('Loading %s to determine feedstock feature size...\n', feedstockFile);
        feedstockDataCheck = load(feedstockFile);
        if isfield(feedstockDataCheck, 'CopyrolysisFeedstockTag')
            FeedType_size = size(feedstockDataCheck.CopyrolysisFeedstockTag, 1);
            fprintf('FeedType_size (from CopyrolysisFeedstockTag rows): %d\n', FeedType_size);
            numBaseFeatures = length(expectedInputCols);
            numFeatures = numBaseFeatures + 2 * FeedType_size;
            fprintf('Total expected feature count: %d (%d base + 2 * %d feedstock)\n', numFeatures, numBaseFeatures, FeedType_size);
        else
            error('CopyrolysisFeedstock.mat does not contain CopyrolysisFeedstockTag field. Cannot determine feature count.');
        end

        numOutputs = numTargetCols;
        fprintf('Using %d outputs for neural network targets (Columns %s).\n', numOutputs, mat2str(expectedTargetCols));

    catch dimErr
        error('Error determining data dimensions: %s\nCheck file paths and formats:\n  Raw Data: %s\n  Feedstock: %s', ...
              dimErr.message, rawDataFile, feedstockFile);
    end

    %% Extract Training Strategy and Ratios from best_config
    fprintf('\n--- Configuring training strategy and data split ---\n');
    strategy = getOrDefault(best_config, 'strategy', 'TT', 'Training Strategy');
    trainRatio = getOrDefault(best_config, 'trainRatio', 0.7, 'Training Ratio');
    if strcmpi(strategy, 'TVT')
        valRatio = getOrDefault(best_config, 'valRatio', 0.15, 'Validation Ratio');
        if valRatio <= 0, warning('Validation Ratio (valRatio=%.2f) must be positive for TVT strategy. Using default 0.15.', valRatio); valRatio = 0.15; end
    else, valRatio = 0; end
    testRatio = 1 - trainRatio - valRatio;
    if trainRatio <= 0 || trainRatio >= 1, error('Invalid trainRatio (%.2f). Must be between 0 and 1.', trainRatio); end
    if valRatio < 0 || valRatio >= 1, error('Invalid valRatio (%.2f). Must be between 0 and 1.', valRatio); end
    if testRatio < 0 || testRatio > 1, error('Invalid calculated testRatio (%.2f). Ratios do not sum to 1 correctly (Train:%.2f, Val:%.2f).', testRatio, trainRatio, valRatio); end
    if abs(trainRatio + valRatio + testRatio - 1.0) > 1e-6, warning('Data split ratios (Train:%.2f, Val:%.2f, Test:%.2f) do not sum exactly to 1. Check configuration.', trainRatio, valRatio, testRatio); end

    fprintf('Model parameters: %d features, %d samples, %d outputs\n', numFeatures, numSamples, numOutputs);
    fprintf('Strategy: %s, TrainRatio: %.2f, ValRatio: %.2f, TestRatio: %.2f\n', strategy, trainRatio, valRatio, testRatio);

    %% Load and Prepare Final Data Matrices
    fprintf('\n--- Loading and processing data for training ---\n');
    try
        % Read numeric data from Excel
        numericData = xlsread(rawDataFile);
        if size(numericData, 1) ~= numSamples
             rangeStr = sprintf('%s%d:%s%d', num2col(1), headerRow + 1, num2col(numCols), numRows);
             fprintf('Warning: xlsread row count (%d) differs from expected (%d). Trying to read range: %s\n', size(numericData,1), numSamples, rangeStr);
              try
                  [~, ~, rawExcelDataRange] = xlsread(rawDataFile, 1, rangeStr);
                  numericData = cell2mat(rawExcelDataRange);
                  if size(numericData,1) ~= numSamples, error('Failed to read correct number of samples even with specific range. Check RawInputData.xlsx structure.'); end
              catch readErrRange, error('Error reading specific range (%s) from RawInputData.xlsx: %s', rangeStr, readErrRange.message); end
         end

        % Extract base features and targets
        inputFeatures = numericData(:, expectedInputCols);
        targetData = numericData(:, expectedTargetCols);
        fprintf('Loaded base features [%d x %d] and targets [%d x %d] from RawInputData.xlsx\n', ...
            size(inputFeatures, 1), size(inputFeatures, 2), size(targetData, 1), size(targetData, 2));

        % Load feedstock data
        fprintf('Loading feedstock variables from %s...\n', feedstockFile);
        feedstockData = load(feedstockFile);
        requiredVars = {'CopyrolysisFeedstockTag', 'Total_FeedID', 'Total_MixingRatio', 'FeedstockIndex'};
        if ~all(isfield(feedstockData, requiredVars)), missingVars = requiredVars(~isfield(feedstockData, requiredVars)); error('Missing required variables in %s: %s', feedstockFile, strjoin(missingVars, ', ')); end
        CopyrolysisFeedstockTag = feedstockData.CopyrolysisFeedstockTag; Total_FeedID = feedstockData.Total_FeedID; Total_MixingRatio = feedstockData.Total_MixingRatio; FeedstockIndex = feedstockData.FeedstockIndex;

        % Verify dimensions of loaded feedstock variables
        if size(Total_FeedID, 1) ~= numSamples || size(Total_MixingRatio, 1) ~= numSamples || size(FeedstockIndex, 1) ~= numSamples
            error('Sample count mismatch between RawInputData (%d) and Feedstock variables (ID:%d, Ratio:%d, Index:%d)', numSamples, size(Total_FeedID, 1), size(Total_MixingRatio, 1), size(FeedstockIndex, 1));
        end
        if size(CopyrolysisFeedstockTag, 1) ~= FeedType_size
             error('Dimension mismatch: CopyrolysisFeedstockTag rows (%d) ~= FeedType_size (%d)', size(CopyrolysisFeedstockTag, 1), FeedType_size);
        end

        % Construct per-sample feedstock features
        fprintf('Constructing per-sample feedstock features using loop...\n');
        MixingFeedID = zeros(numSamples, FeedType_size); MixingRatio = zeros(numSamples, FeedType_size); validSampleCount = 0;
        for i = 1 : numSamples
            idx1 = FeedstockIndex(i, 1); idx2 = FeedstockIndex(i, 2);
            isValidIdx1 = isnumeric(idx1) && isscalar(idx1) && isfinite(idx1) && idx1 >= 1 && idx1 <= FeedType_size && (idx1 == floor(idx1));
            isValidIdx2 = isnumeric(idx2) && isscalar(idx2) && isfinite(idx2) && idx2 >= 1 && idx2 <= FeedType_size && (idx2 == floor(idx2));
            if ~isValidIdx1 || ~isValidIdx2, warning('Sample %d has invalid FeedstockIndex [%s, %s]. Indices must be integers between 1 and %d. Skipping sample features construction.', i, num2str(idx1), num2str(idx2), FeedType_size); continue; end
            MixingFeedID(i, idx1) = Total_FeedID(i, 1); MixingFeedID(i, idx2) = Total_FeedID(i, 2); MixingRatio(i, idx1) = Total_MixingRatio(i, 1); MixingRatio(i, idx2) = Total_MixingRatio(i, 2); validSampleCount = validSampleCount + 1;
        end
        if validSampleCount ~= numSamples, fprintf('Warning: Features constructed for %d out of %d samples due to invalid FeedstockIndex values.\n', validSampleCount, numSamples); end
        Feedstock4training = [MixingFeedID MixingRatio];
        fprintf('Constructed Feedstock4training matrix [%d x %d]\n', size(Feedstock4training,1), size(Feedstock4training,2));

        % Concatenate features
        X = [inputFeatures Feedstock4training];
        fprintf('Final input matrix X created by concatenation [%d x %d]\n', size(X,1), size(X,2));
        if size(X, 2) ~= numFeatures, error('CRITICAL: Final input matrix columns (%d) do not match expected feature count (%d). Check processing logic.', size(X, 2), numFeatures); end
        y = targetData;

    catch loadErr
        error('Error loading or processing data for training: %s', loadErr.message);
    end

    % --- Get Preprocessing Settings BEFORE training ---
    fprintf('Obtaining preprocessing settings (PS/TS)...\n');
    try
        % Calculate PS/TS directly using MATLAB's functions on the full dataset
        % Use documented syntax: [Y, PS] = mapminmax(X) [cite: 2, 17]
        % Data is expected as Features x Samples for mapminmax, so transpose X and y.
        % Capture BOTH output arguments to avoid MATLAB:unassignedOutputs error.
        
        % ***** CORRECTED LINES based on Documentation & Error Log *****
        [Xnorm_temp, PS] = mapminmax(X'); % Capture 1st output (Y) and 2nd (PS)
        [ynorm_temp, TS] = mapminmax(y'); % Capture 1st output (Y) and 2nd (TS)
        % *************************************************************

        fprintf('Calculated PS and TS using mapminmax directly on full dataset.\n');

        PS_global = PS; % Keep global copies if needed elsewhere
        TS_global = TS;
    catch preprocSetErr
        fprintf('ERROR obtaining preprocessing settings before training: %s\n', preprocSetErr.message);
        % Create empty/error placeholders if generation fails
        PS = createPlaceholderPS(); PS.status = ['Preprocessing settings generation failed: ' preprocSetErr.message];
        TS = createPlaceholderTS(); TS.status = ['Preprocessing settings generation failed: ' preprocSetErr.message];
        PS_global = PS; TS_global = TS;
        if isempty(e), e = preprocSetErr; end % Store error
        % Rethrow the error to stop execution if PS/TS are critical
        rethrow(preprocSetErr); 
    end
    % --- End Preprocessing Settings ---

    % Convert data to format expected by NN functions (Features x Samples)
    input_data = X';  
    target_data = y'; 

    fprintf('Final data dimensions prepared for NN training:\n');
    fprintf('- input_data (Features x Samples): [%d x %d]\n', size(input_data, 1), size(input_data, 2));
    fprintf('- target_data (Outputs x Samples): [%d x %d]\n', size(target_data, 1), size(target_data, 2));
    if size(input_data, 1) ~= numFeatures || size(target_data, 1) ~= numOutputs || size(input_data, 2) ~= numSamples || size(target_data, 2) ~= numSamples
        error('CRITICAL dimension mismatch between calculated parameters and final data matrices. Check data loading/processing.');
    end

    %% Configure Neural Network
    fprintf('\n--- Configuring neural network based on best hyperparameters ---\n');
    % --- Get Hyperparameters from best_config ---
    % Use getOrDefault to handle potentially missing fields and assign defaults
    config_lr = getOrDefault(best_config, 'lr', 0.01, 'Learning Rate (lr)');
    config_mc = getOrDefault(best_config, 'mc', 0.7, 'Momentum (mc)');
    config_hiddenLayer = getOrDefault(best_config, 'hiddenLayer', [10], 'Hidden Layer Structure');
    config_hiddenTF = getOrDefault(best_config, 'hiddenTF', 'poslin', 'Hidden Transfer Function');
    % Fallback for older config files that might use 'transferFcn'
    if ~isfield(best_config, 'hiddenTF') && isfield(best_config, 'transferFcn')
        config_hiddenTF = getOrDefault(best_config, 'transferFcn', 'poslin', 'Transfer Function (fallback)');
        fprintf('INFO: Using ''transferFcn'' field from best_config as original hiddenTF.\n');
    end
    config_outputTF = getOrDefault(best_config, 'outputTF', 'purelin', 'Output Transfer Function');
    config_lr_inc = getOrDefault(best_config, 'lr_inc', 1.05, 'Learning Rate Increase');
    config_lr_dec = getOrDefault(best_config, 'lr_dec', 0.7, 'Learning Rate Decrease');
    % --- End Get Hyperparameters ---

    % --- Apply Overrides for Stability ---
    % Change activation function for potentially better stability
    final_hiddenTF = 'tansig'; % Override to tansig
    % Use a conservative starting LR
    final_initial_lr = 0.01;
    % Reduce momentum
    final_mc = 0.5;
    % Use original dynamic factors from optimization for nnbp logic
    final_lr_inc = config_lr_inc;
    final_lr_dec = config_lr_dec;

    fprintf('INFO: Original best config HiddenTF=%s, LR=%.4f, MC=%.2f\n', config_hiddenTF, config_lr, config_mc);
    fprintf('INFO: Overriding for stability: HiddenTF=%s, Initial LR=%.4f, MC=%.2f\n', final_hiddenTF, final_initial_lr, final_mc);
    fprintf('INFO: Using dynamic LR factors from config: LR_INC=%.4f, LR_DEC=%.4f\n', final_lr_inc, final_lr_dec);
    % --- End Overrides ---

    % Ensure hiddenLayer is a row vector
    if isscalar(config_hiddenLayer), config_hiddenLayer = [config_hiddenLayer]; elseif ~isrow(config_hiddenLayer), config_hiddenLayer = config_hiddenLayer(:)'; end

    % Create the network using the potentially overridden parameters
    fprintf('DEBUG: Calling nncreate with numFeatures=%d, hiddenLayer=[%s], numOutputs=%d, hiddenTF=%s, outputTF=%s\n', ...
            numFeatures, num2str(config_hiddenLayer), numOutputs, final_hiddenTF, config_outputTF);
    try
        net = nncreate(numFeatures, config_hiddenLayer, numOutputs, final_hiddenTF, config_outputTF);
        fprintf('DEBUG: nncreate completed. Network structure initialized.\n');
    catch createErr
        error('Failed to create network with nncreate: %s', createErr.message);
    end

    %% Set Training Parameters in Network Structure
    fprintf('\n--- Setting network training parameters ---\n');
    net.performFcn = 'mse';
    net.divideFcn = 'dividerand'; % Ensure data division happens correctly
    net.divideParam.trainRatio = trainRatio;
    net.divideParam.valRatio = valRatio;
    net.divideParam.testRatio = testRatio;
    fprintf('Data division ratios set: Train=%.2f, Val=%.2f, Test=%.2f\n', net.divideParam.trainRatio, net.divideParam.valRatio, net.divideParam.testRatio);

    net.trainParam.epochs = 5000;
    net.trainParam.goal = 1e-6;
    net.trainParam.min_grad = 1e-8;
    net.trainParam.max_fail = 25; % Used only if validation is enabled
    net.trainParam.showWindow = false;
    net.trainParam.showCommandLine = true;
    net.trainParam.show = 25; % Frequency of command line updates

    % --- Assign final training parameters to the net structure ---
    net.trainParam.lr = final_initial_lr; % Use the overridden initial LR
    net.trainParam.mc = final_mc;         % Use the overridden momentum
    net.trainParam.lr_inc = final_lr_inc; % Use original factor (nnbp controls extremes)
    net.trainParam.lr_dec = final_lr_dec; % Use original factor (nnbp controls extremes)
    % --- End Assign final params ---

    fprintf('Final Training parameters set: Epochs=%d, Goal=%.e, Initial LR=%.4f, MC=%.2f, LR_INC=%.2f, LR_DEC=%.2f\n', ...
            net.trainParam.epochs, net.trainParam.goal, net.trainParam.lr, net.trainParam.mc, net.trainParam.lr_inc, net.trainParam.lr_dec);
    %% Train the Neural Network
    fprintf('\n--- Training neural network using %s strategy ---\n', upper(strategy));
    fprintf('DEBUG: Checking final dimensions before calling nntrain...\n');
    fprintf('DEBUG: Size of input_data: [%d x %d]\n', size(input_data, 1), size(input_data, 2)); fprintf('DEBUG: Size of target_data: [%d x %d]\n', size(target_data, 1), size(target_data, 2));
    fprintf('DEBUG: Network expected input size (net.numInput): %d\n', net.numInput); fprintf('DEBUG: Network expected output size (net.numOutput): %d\n', net.numOutput);
    if ~isfield(net,'numInput') || net.numInput ~= size(input_data, 1) || ~isfield(net,'numOutput') || net.numOutput ~= size(target_data, 1), error('CRITICAL dimension mismatch between network structure and prepared data. Check nncreate or data processing.'); end

    tr = struct(); % Initialize training record
    try
        fprintf('DEBUG: Calling custom nntrain function...\n');
        [net, tr] = nntrain(net, input_data, target_data);
        fprintf('DEBUG: nntrain call completed.\n');
        if isfield(tr, 'stop') && ~isempty(tr.stop), fprintf('INFO: Training stopped. Reason: %s\n', tr.stop{end}); else fprintf('INFO: Training completed (or tr.stop field missing).\n'); end
        if isfield(tr, 'best_epoch') && ~isempty(tr.best_epoch), fprintf('INFO: Best performance achieved at epoch %d.\n', tr.best_epoch(end)); end
    catch trainError
        fprintf('\n===== FATAL ERROR during nntrain =====\n'); fprintf('ERROR MESSAGE: %s\n', trainError.message); fprintf('ERROR IDENTIFIER: %s\n', trainError.identifier); fprintf('STACK TRACE:\n');
        for k=1:length(trainError.stack), fprintf('  File: %s, Name: %s, Line: %d\n', trainError.stack(k).file, trainError.stack(k).name, trainError.stack(k).line); end
        fprintf('======================================\n');
        tr = createMinimalTR(trainError.message, 'Failed in nntrain'); fprintf('Attempting to continue to save results despite nntrain error...\n'); e = trainError;
    end

    %% Evaluate Performance
    fprintf('\n--- Evaluating final model performance ---\n');
    trainPerf = NaN; valPerf = NaN; testPerf = NaN;
    if exist('tr', 'var') && isstruct(tr) && ~isempty(fieldnames(tr))
        if isfield(tr, 'status')
            if iscell(tr.status) && ~isempty(tr.status)
                % take the first element if it is a cell array
                statusStr = tr.status{1};
                if contains(statusStr, 'Failed', 'IgnoreCase', true)
                    fprintf('Training status in ''tr'' indicates failure: "%s". Performance metrics may be NaN.\n', statusStr);
                end
            elseif ischar(tr.status) && contains(tr.status, 'Failed', 'IgnoreCase', true)
                % if it is a string
                fprintf('Training status in ''tr'' indicates failure: "%s". Performance metrics may be NaN.\n', tr.status);
            end
        end
        
        % Check performance metrics
        if isfield(tr, 'best_perf') && isscalar(tr.best_perf) && isnumeric(tr.best_perf) && isfinite(tr.best_perf) && ~isnan(tr.best_perf)
            trainPerf = tr.best_perf;
            fprintf('  Best Training MSE (tr.best_perf):   %.6f\n', trainPerf);
        else
            fprintf('  Best Training MSE: Not available, NaN, or Inf in training record (tr.best_perf).\n');
        end

        if isfield(tr, 'best_vperf') && isscalar(tr.best_vperf) && isnumeric(tr.best_vperf) && ~isnan(tr.best_vperf) && isfinite(tr.best_vperf)
            valPerf = tr.best_vperf; 
            fprintf('  Best Validation MSE (tr.best_vperf): %.6f\n', valPerf); 
        else 
            fprintf('  Best Validation MSE: Not available or not applicable (tr.best_vperf).\n'); 
        end
        
        if isfield(tr, 'best_tperf') && isscalar(tr.best_tperf) && isnumeric(tr.best_tperf) && ~isnan(tr.best_tperf) && isfinite(tr.best_tperf)
            testPerf = tr.best_tperf; 
            fprintf('  Best Testing MSE (tr.best_tperf):    %.6f\n', testPerf); 
        else 
            fprintf('  Best Testing MSE: Not available in training record (tr.best_tperf).\n'); 
        end
    else
        fprintf('Training record structure ''tr'' is missing, empty, or invalid. Cannot report performance.\n');
    end

    %% Save the Trained Model and Results
    fprintf('\n--- Saving trained model and results ---\n');
    fprintf('Using preprocessing structures PS and TS obtained before training.\n');
    if ~exist('PS','var') || ~isstruct(PS) || ~exist('TS','var') || ~isstruct(TS)
         warning('Preprocessing structures PS/TS not found before save. Saving placeholders.');
         PS = createPlaceholderPS(); PS.status = 'PS not found before save.';
         TS = createPlaceholderTS(); TS.status = 'TS not found before save.';
    end
    if ~exist('PS_global','var'), PS_global = PS; end 
    if ~exist('TS_global','var'), TS_global = TS; end 

    params = struct();
    % params.lr = lr; params.mc = mc; params.lr_inc = lr_inc; params.lr_dec = lr_dec; params.hiddenLayer = hiddenLayer; params.hiddenTF = hiddenTF; params.outputTF = outputTF;
    params.lr = final_initial_lr; 
    params.mc = final_mc;  
    params.lr_inc = final_lr_inc; 
    params.lr_dec = final_lr_dec; 
    params.hiddenLayer = config_hiddenLayer;
    params.hiddenTF = final_hiddenTF;
    params.outputTF = config_outputTF;
    params.strategy = strategy; params.trainRatio = trainRatio; params.valRatio = valRatio; params.testRatio = testRatio;
    if isfield(net, 'trainParam'), params.epochs = net.trainParam.epochs; params.goal = net.trainParam.goal; params.min_grad = net.trainParam.min_grad; params.max_fail = net.trainParam.max_fail; else params.epochs = NaN; end
    params.finalTrainMSE = trainPerf; params.finalValMSE = valPerf; params.finalTestMSE = testPerf;
    fprintf('Parameter structure for saving created.\n'); disp(params);

    if ~exist('tr', 'var') || ~isstruct(tr) || isempty(fieldnames(tr))
        fprintf('Warning: Training record ''tr'' not found or invalid before final save. Using placeholder.\n');
        errorMessage = 'Training did not run or tr not generated/valid'; if ~isempty(e), errorMessage = e.message; end
        tr = createMinimalTR(errorMessage, 'Not Run/Failed Early');
    end
    if ~isfield(tr, 'status'), if ~isempty(e), tr.status = ['Failed: ' e.message]; else tr.status = 'Completed or Failed - Status Unknown'; end; end

    % Create trainResults structure for model validation in complete_workflow.m
    trainResults = struct();
    trainResults.best_perf = NaN;
    if isfield(tr, 'best_perf') && ~isnan(tr.best_perf), trainResults.best_perf = tr.best_perf; end
    if isfield(tr, 'best_vperf') && ~isnan(tr.best_vperf), trainResults.best_vperf = tr.best_vperf; end
    if isfield(tr, 'best_tperf') && ~isnan(tr.best_tperf), trainResults.best_tperf = tr.best_tperf; end
    if isfield(tr, 'status'), trainResults.status = tr.status; end
    
    % Add neuronsInLayers directly to trainResults for complete_workflow.m to use
    % This avoids the need to access net.layers which doesn't exist (it's net.layer)
    if isfield(net, 'layer') && ~isempty(net.layer)
        try
            % Extract neuron counts from each layer and store them
            neuronsInLayers = cell(1, length(net.layer));
            for i = 1:length(net.layer)
                if isfield(net.layer{i}, 'size') && ~isempty(net.layer{i}.size)
                    neuronsInLayers{i} = net.layer{i}.size;
                else
                    neuronsInLayers{i} = 0; % Default if size not found
                end
            end
            trainResults.neuronsInLayers = neuronsInLayers;
            
            % Add an actual 'layers' field with the same content to maintain compatibility
            % with any code that might be looking for this field name
            trainResults.layers = neuronsInLayers;
        catch layer_err
            fprintf('Warning: Could not extract layer neurons: %s\n', layer_err.message);
            trainResults.neuronsInLayers = {0}; % Default fallback
            trainResults.layers = {0}; % Also set the layers field for compatibility
        end
    else
        trainResults.neuronsInLayers = {0}; % Default if no layers
        trainResults.layers = {0}; % Also set the layers field for compatibility
    end
    
    fprintf('Created trainResults structure for model validation.\n');

    fprintf('Saving final results to .mat files...\n');
    bestModelFile = fullfile(bestModelDir, 'best_model.mat');
    save(bestModelFile, 'net', 'tr', 'trainResults', 'params', 'input_data', 'target_data', 'PS', 'TS', 'PS_global', 'TS_global', 'numFeatures', 'numOutputs', 'numSamples', 'FeedType_size');
    fprintf('Best model results saved to: %s\n', bestModelFile);

    trainingModelFile = fullfile(trainingDir, 'trained_model.mat');
    save(trainingModelFile, 'net', 'tr', 'trainResults', 'params', 'input_data', 'target_data', 'PS', 'TS', 'PS_global', 'TS_global', 'numFeatures', 'numOutputs', 'numSamples', 'FeedType_size');
    fprintf('Training model copy saved to: %s\n', trainingModelFile);

    executionTime = toc(startTime);
    fprintf('\nTotal execution time for train_best_model: %.2f seconds (%.2f minutes)\n', executionTime, executionTime/60);
    fprintf('\nTrain best model script finished.\n');

catch e % Catch block for the main script try
    fprintf('\n===== ERROR OCCURRED in train_best_model.m =====\n'); fprintf('Error: %s\n', e.message); fprintf('Identifier: %s\n', e.identifier); fprintf('Stack trace:\n'); disp(e.stack);
    fprintf('\nAttempting to save basic results despite error...\n');
    try
        if ~exist('rootDir','var') || isempty(rootDir), try scriptPath = mfilename('fullpath'); scriptDir = fileparts(scriptPath); rootDir = fileparts(fileparts(scriptDir)); catch, rootDir = pwd; fprintf('Warning: Could not determine rootDir automatically, using pwd: %s\n', rootDir); end; end
        if ~exist('resultsDir','var') || isempty(resultsDir), resultsDir = fullfile(rootDir, 'results'); end; if ~exist('bestModelDir','var') || isempty(bestModelDir), bestModelDir = fullfile(resultsDir, 'best_model'); end
        if ~exist(resultsDir, 'dir'), mkdir(resultsDir); end; if ~exist(bestModelDir, 'dir'), mkdir(bestModelDir); end
        essentialsExist = exist('net','var') && isstruct(net) && exist('input_data','var') && ~isempty(input_data) && exist('target_data','var') && ~isempty(target_data);
        if essentialsExist
            fprintf('Essential variables (net, input_data, target_data) exist. Creating placeholders for others...\n');
            if ~exist('PS', 'var') || ~isstruct(PS), PS = createPlaceholderPS(); PS.status = ['Emergency save: ' e.message]; end; if ~exist('TS', 'var') || ~isstruct(TS), TS = createPlaceholderTS(); TS.status = ['Emergency save: ' e.message]; end
            PS_global = PS; TS_global = TS;
            if ~exist('params', 'var') || ~isstruct(params), params = createPlaceholderParams(); end; params.status = ['Failed in train_best_model: ' e.message]; params.error_identifier = e.identifier;
            tr_emergency = createMinimalTR(e.message, ['Failed: ' e.identifier]);
            
            % Create trainResults structure for emergency save
            trainResults_emergency = struct();
            trainResults_emergency.best_perf = NaN;
            trainResults_emergency.status = ['Failed: ' e.identifier];
            trainResults_emergency.error_message = e.message;
            
            % Add neuronsInLayers to emergency trainResults structure
            if isfield(net, 'layer') && ~isempty(net.layer)
                try
                    % Extract neuron counts from each layer
                    neuronsInLayers_emergency = cell(1, length(net.layer));
                    for i = 1:length(net.layer)
                        if isfield(net.layer{i}, 'size') && ~isempty(net.layer{i}.size)
                            neuronsInLayers_emergency{i} = net.layer{i}.size;
                        else
                            neuronsInLayers_emergency{i} = 0; % Default if size not found
                        end
                    end
                    trainResults_emergency.neuronsInLayers = neuronsInLayers_emergency;
                    
                    % Add an actual 'layers' field for compatibility
                    trainResults_emergency.layers = neuronsInLayers_emergency;
                catch layer_err
                    fprintf('Warning: Could not extract layer neurons for emergency save: %s\n', layer_err.message);
                    trainResults_emergency.neuronsInLayers = {0}; % Default fallback
                    trainResults_emergency.layers = {0}; % Also set the layers field for compatibility
                end
            else
                trainResults_emergency.neuronsInLayers = {0}; % Default if no layers
                trainResults_emergency.layers = {0}; % Also set the layers field for compatibility
            end
            
            if exist('numFeatures','var'), params.numFeatures = numFeatures; else params.numFeatures = NaN; end; if exist('numOutputs','var'), params.numOutputs = numOutputs; else params.numOutputs = NaN; end
            if exist('numSamples','var'), params.numSamples = numSamples; else params.numSamples = NaN; end; if exist('FeedType_size','var'), params.FeedType_size = FeedType_size; else params.FeedType_size = NaN; end
            failedModelFile = fullfile(bestModelDir, 'best_model_FAILED.mat');
            save(failedModelFile, 'net', 'tr_emergency', 'trainResults_emergency', 'params', 'input_data', 'target_data', 'PS', 'TS', 'PS_global', 'TS_global');
            fprintf('Emergency results saved to: %s\n', failedModelFile);
        else fprintf('ERROR: Missing essential variables (net, input_data, target_data). Cannot save emergency results.\n'); end
    catch saveErr, fprintf('ERROR: Failed during attempt to save emergency results: %s\n', saveErr.message); end
end % End main try-catch block


%% --- Completion Marker Logic ---
finalPerformance = NaN; % Initialize
fprintf('\n--- Checking final results for completion marker ---\n');
scriptErrorOccurred = ~isempty(e); % Check if an error was caught by the main try-catch
trExists = exist('tr', 'var') && isstruct(tr) && ~isempty(fieldnames(tr)); % Check if tr exists and is valid

trainingSucceeded = false; % Default assumption

if ~scriptErrorOccurred && trExists
    fprintf('Script finished without catching errors. Training record ''tr'' exists.\n');
    % Check status field first
    if isfield(tr, 'status')
        if iscell(tr.status) && ~isempty(tr.status)
            statusStr = tr.status{1};
            if contains(statusStr, 'Failed', 'IgnoreCase', true)
                fprintf('Training status in ''tr'' indicates failure: "%s".\n', statusStr);
            end
        elseif ischar(tr.status) && contains(tr.status, 'Failed', 'IgnoreCase', true)
            fprintf('Training status in ''tr'' indicates failure: "%s".\n', tr.status);
        end
    elseif isfield(tr, 'best_perf') && isscalar(tr.best_perf) && isnumeric(tr.best_perf) && ~isnan(tr.best_perf) && ~isinf(tr.best_perf)
        finalPerformance = tr.best_perf;
        trainingSucceeded = true; % Mark as succeeded only if perf is a valid finite number
        fprintf('Final training performance (best_perf) from ''tr'': %.6e\n', finalPerformance);
    else
        fprintf('Warning: tr.best_perf field missing, empty, invalid, NaN or Inf in final ''tr''. Assuming training failure for marker.\n');
    end
else
    fprintf('Script finished with errors OR final training record ''tr'' is invalid/missing.\n');
    if scriptErrorOccurred, fprintf('Error caught: %s\n', e.message); end
end

% Determine marker status and path
if ~exist('bestModelDir','var') || isempty(bestModelDir)
    try % Try to determine path again just in case
        scriptPath = mfilename('fullpath'); scriptDir = fileparts(scriptPath); rootDir = fileparts(fileparts(scriptDir));
    catch, rootDir = pwd; end
    resultsDir = fullfile(rootDir, 'results'); bestModelDir = fullfile(resultsDir, 'best_model');
end
resultPathSuccess = fullfile(bestModelDir, 'best_model.mat');
resultPathFailed = fullfile(bestModelDir, 'best_model_FAILED.mat');

% Prepare metadata
metadata = struct('finalPerformance', finalPerformance);
if scriptErrorOccurred
    metadata.status = 'Failed';
    metadata.error_message = e.message;
    metadata.error_identifier = e.identifier;
    marker_status = 'training_failed';
    marker_path = resultPathFailed; % Point to the FAILED file if it exists
    if ~exist(marker_path, 'file'), marker_path = resultPathSuccess; end % Fallback if FAILED file wasn't saved
    fprintf('Creating FAILED marker, referencing: %s\n', marker_path);
else % Script didn't error, but training might have failed internally
    if trainingSucceeded && exist(resultPathSuccess, 'file')
         metadata.status = 'Completed';
         marker_status = 'training';
         marker_path = resultPathSuccess;
         fprintf('Creating SUCCESS marker for: %s\n', marker_path);
    else % Training failed (e.g., NaN perf) or success file missing
        metadata.status = 'CompletedWithTrainingIssue';
        metadata.warning = 'Script completed, but training failed/stalled or output file missing.';
         if isfield(tr, 'status') && ~isempty(tr.status), metadata.tr_status = strjoin(tr.status,'; '); end
         marker_status = 'training_failed';
         marker_path = resultPathFailed;
         if ~exist(marker_path, 'file'), marker_path = resultPathSuccess; end
         fprintf('Creating FAILED marker (due to training issue), referencing: %s\n', marker_path);
    end
end

fprintf('Metadata for completion marker:\n'); disp(metadata);
diary off; % Turn off diary before creating marker

% Create the marker
if exist('create_completion_marker', 'file') == 2
    fprintf('Attempting to create completion marker...\n');
    try
        create_completion_marker(marker_status, marker_path, metadata);
        fprintf('✓ Completion marker created (Status: %s).\n', marker_status);
    catch markerErr
        fprintf('ERROR creating completion marker: %s\n', markerErr.message);
        diary on; % Turn diary back on if marker fails
    end
else
    fprintf('Note: create_completion_marker function not found. Skipping marker creation.\n');
end

%% --- Final Exit ---
if ~isempty(e), fprintf('\nExiting train_best_model.m due to captured error.\n'); % exit(1); % Uncomment for non-zero exit
end
fprintf('\nExiting train_best_model.m normally.\n');

%% ===== HELPER FUNCTIONS =====
function value = getOrDefault(structVar, fieldName, defaultValue, fieldDesc)
    if nargin < 4, fieldDesc = fieldName; end
    if isstruct(structVar) && isfield(structVar, fieldName) && ~isempty(structVar.(fieldName)), value = structVar.(fieldName); else value = defaultValue; if ~isstruct(structVar) || ~isfield(structVar, fieldName) || isempty(structVar.(fieldName)), fprintf('Warning: Parameter "%s" (%s) not found or empty in provided structure. Using default value: %s\n', fieldDesc, fieldName, mat2str(value)); end; end
end
function tr = createMinimalTR(errorMessage, statusMessage)
    if nargin < 2 || isempty(statusMessage), statusMessage = 'Failed'; end; if nargin < 1 || isempty(errorMessage), errorMessage = 'Unknown error during training'; end
    tr = struct('trainInd', [], 'valInd', [], 'testInd', [], 'stop', {{errorMessage}}, 'status', statusMessage, 'best_epoch', NaN, 'goal', NaN, 'states', {{}}, 'best_perf', NaN, 'best_vperf', NaN, 'best_tperf', NaN, 'epoch', NaN, 'time', NaN, 'perf', NaN, 'vperf', NaN, 'tperf', NaN, 'gradient', NaN, 'num_epochs', NaN, 'val_fail', NaN);
end
function PS = createPlaceholderPS(), PS = struct('name', 'placeholder', 'status', 'Created during emergency save', 'xoffset', [], 'gain', [], 'ymin', -1); end
function TS = createPlaceholderTS(), TS = struct('name', 'placeholder', 'status', 'Created during emergency save', 'xoffset', [], 'gain', [], 'ymin', -1); end
function params = createPlaceholderParams(), params = struct('lr', NaN, 'mc', NaN, 'hiddenLayer', [], 'hiddenTF', 'Unknown', 'outputTF', 'Unknown', 'strategy', 'Unknown', 'status', 'Created during emergency save', 'finalTrainMSE', NaN, 'finalValMSE', NaN, 'finalTestMSE', NaN); end
function msg = safeGetErrorMsg(errVar), if exist('errVar','var') && isobject(errVar) && isprop(errVar, 'message') && ~isempty(errVar.message), msg = errVar.message; else msg = 'Unknown error or error occurred before variable capture.'; end; end
function colStr = num2col(colNum), if ~isnumeric(colNum) || ~isscalar(colNum) || colNum < 1 || colNum ~= floor(colNum), error('Input must be a positive integer.'); end; A = 'A'; Z = 'Z'; base = Z-A+1; colStr = ''; while colNum > 0, rem = mod(colNum-1, base); colStr = [char(A+rem), colStr]; colNum = floor((colNum - rem - 1) / base); end; end
function result = iif(condition, trueVal, falseVal), if condition, result = trueVal; else result = falseVal; end; end