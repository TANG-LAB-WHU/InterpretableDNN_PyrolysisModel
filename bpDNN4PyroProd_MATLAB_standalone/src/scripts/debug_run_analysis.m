%% Debug run analysis script - simplified version
fprintf('Starting debug_run_analysis.m\n');

% Setup output log file
diary('debug_run_analysis_log.txt');
fprintf('Logging to debug_run_analysis_log.txt\n');

try
    % Get the root directory of the project
    rootDir = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    fprintf('Root directory: %s\n', rootDir);
    
    % Set up paths - only add directories that exist
    directories_to_add = {
        fullfile(rootDir, 'src', 'shap'),
        fullfile(rootDir, 'src', 'dnn'),
        fullfile(rootDir, 'src', 'utils'),
        fullfile(rootDir, 'src', 'scripts'),
        fullfile(rootDir, 'src', 'visualization')
    };
    
    for i = 1:length(directories_to_add)
        if exist(directories_to_add{i}, 'dir')
            addpath(directories_to_add{i});
            fprintf('Added to path: %s\n', directories_to_add{i});
        else
            fprintf('Directory does not exist (skipping): %s\n', directories_to_add{i});
        end
    end
    fprintf('Paths setup complete.\n');
    
    % Define directory paths relative to root directory
    scriptDir = fullfile(rootDir, 'src', 'scripts');
    shapDir = fullfile(rootDir, 'src', 'shap');
    trainingDir = fullfile(rootDir, 'debug_dir'); % For debug output
    modelDir = fullfile(rootDir, 'src', 'model'); % Use src/model instead of model_data
    bestModelDir = fullfile(rootDir, 'results', 'best_model'); % Best model directory
    optimizationDir = fullfile(rootDir, 'results', 'optimization'); % Optimization directory
    analysisDir = fullfile(rootDir, 'results', 'analysis', 'debug');
    
    % Create the data directories
    analysisDataDir = fullfile(analysisDir, 'data');
    analysisFigDir = fullfile(analysisDir, 'figures');
    
    % Check if directories exist, create if they don't
    dirs_to_check = {};
    dirs_to_check{1} = trainingDir;
    dirs_to_check{2} = bestModelDir;
    dirs_to_check{3} = optimizationDir;
    dirs_to_check{4} = analysisDir;
    dirs_to_check{5} = analysisDataDir;
    dirs_to_check{6} = analysisFigDir;
    
    for i = 1:length(dirs_to_check)
        if ~exist(dirs_to_check{i}, 'dir')
            fprintf('Creating directory: %s\n', dirs_to_check{i});
            mkdir(dirs_to_check{i});
        else
            fprintf('Directory exists: %s\n', dirs_to_check{i});
        end
    end
    
    % Add model directory to path if it's not already
    if ~contains(path, modelDir)
        addpath(modelDir);
        fprintf('Added model directory to path: %s\n', modelDir);
    end
    
    % Create a simple test model data file
    testModelFile = fullfile(trainingDir, 'Results_trained.mat');
    if ~exist(testModelFile, 'file')
        fprintf('Creating test model file: %s\n', testModelFile);
        
        % Create simple test data
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
        
        % Create preprocessing structures
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
        
        % Save the data
        save(testModelFile, 'input', 'target', 'net', 'PS', 'TS', 'PS_global', 'TS_global');
        fprintf('Saved test model to: %s\n', testModelFile);
    else
        fprintf('Test model file already exists: %s\n', testModelFile);
    end
    
    % Skip hyperparameter optimization for debug mode
    fprintf('\n===== SKIPPING HYPERPARAMETER OPTIMIZATION FOR DEBUG MODE =====\n');
    
    % Run the SHAP analysis
    fprintf('\n===== RUNNING SIMPLIFIED SHAP ANALYSIS (DEBUG MODE) =====\n');
    
    % Create simplified SHAP values
    fprintf('Generating simplified SHAP values...\n');
    
    % Load the test model data
    fprintf('Loading test model data from: %s\n', testModelFile);
    data = load(testModelFile);
    
    % Extract data components
    input = data.input;
    target = data.target;
    net = data.net;
    PS = data.PS;
    TS = data.TS;
    
    % Create SHAP values - simplified approach
    [numFeatures, numSamples] = size(input);
    [numOutputs, ~] = size(target);
    
    % Initialize SHAP values
    shapValues = zeros(numSamples, numFeatures, numOutputs);
    baseValue = mean(target, 2)'; % Use mean of target as base value
    
    % Create feature names
    varNames = cell(numFeatures, 1);
    for i = 1:numFeatures
        varNames{i} = sprintf('Feature_%d', i);
    end
    
    % Simplified prediction function for testing
    nnpredict = @(net, x) ones(numOutputs, 1) .* rand(numOutputs, 1); % Dummy prediction that returns a vector of length numOutputs
    
    % Generate SHAP values
    for i = 1:numSamples
        for j = 1:numOutputs
            % Get a dummy prediction for this sample
            sample_pred = nnpredict(net, input(:, i));
            sample_pred = sample_pred(j);
            
            % Calculate the difference from base value
            diff_from_base = sample_pred - baseValue(j);
            
            % Distribute this difference among features as SHAP values
            feature_importances = rand(1, numFeatures);
            feature_importances = feature_importances / sum(feature_importances);
            shapValues(i, :, j) = diff_from_base * feature_importances;
        end
        
        % Report progress
        if mod(i, 5) == 0 || i == numSamples
            fprintf('Processed %d/%d samples\n', i, numSamples);
        end
    end
    
    % Save SHAP results
    fprintf('Saving SHAP results...\n');
    resultFile = fullfile(analysisDataDir, 'shap_results.mat');
    save(resultFile, 'shapValues', 'baseValue', 'varNames', 'input', 'target');
    fprintf('SHAP results saved to: %s\n', resultFile);
    
    fprintf('\n===== DEBUG RUN ANALYSIS COMPLETE =====\n');
    fprintf('Saved SHAP results to %s\n', resultFile);
catch e
    % Handle errors
    fprintf('\n===== ERROR OCCURRED =====\n');
    fprintf('Error: %s\n', e.message);
    fprintf('Stack trace:\n');
    disp(e.stack);
end

% Close diary
diary off; 