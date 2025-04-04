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
    trainingDir = fullfile(rootDir, 'results', 'debug'); % Use results/debug directory
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
        % Load the test model data to use for optimization
        data = load(testModelFile);
        input = data.input;
        target = data.target;
    end
    
    % Run a simplified hyperparameter optimization for debug mode
    fprintf('\n===== RUNNING SIMPLIFIED HYPERPARAMETER OPTIMIZATION (DEBUG MODE) =====\n');
    
    % Create optimization log file
    optLogFile = fullfile(optimizationDir, 'debug_optimization_log.txt');
    if exist(optLogFile, 'file')
        delete(optLogFile);
    end
    optLogFid = fopen(optLogFile, 'w');
    fprintf(optLogFid, 'Debug Hyperparameter Optimization Log\n');
    fprintf(optLogFid, 'Date: %s\n\n', datestr(now));
    
    % Define a small set of hyperparameters to test (simplified for debug)
    lrValues = [0.01, 0.1];         % Learning rates
    mcValues = [0.8, 0.9];          % Momentum coefficients
    hiddenLayerValues = {
        [10],                      % 1 hidden layer with 10 neurons
        [8, 5]                     % 2 hidden layers
    };
    hiddenTFValues = {'tansig', 'logsig'};  % Transfer functions
    
    % Calculate total configurations
    totalConfigs = length(lrValues) * length(mcValues) * length(hiddenLayerValues) * length(hiddenTFValues);
    fprintf('Testing %d hyperparameter configurations in debug mode\n', totalConfigs);
    fprintf(optLogFid, 'Testing %d hyperparameter configurations\n', totalConfigs);
    
    % Initialize best model parameters
    bestMSE = Inf;
    bestConfig = struct();
    bestNetFile = fullfile(optimizationDir, 'debug_best_model.mat');
    
    % Run all configurations
    configCounter = 0;
    for lr_idx = 1:length(lrValues)
        for mc_idx = 1:length(mcValues)
            for hl_idx = 1:length(hiddenLayerValues)
                for tf_idx = 1:length(hiddenTFValues)
                    % Current configuration
                    configCounter = configCounter + 1;
                    fprintf('Configuration %d/%d:\n', configCounter, totalConfigs);
                    
                    % Extract parameters
                    lr = lrValues(lr_idx);
                    mc = mcValues(mc_idx);
                    hiddenLayers = hiddenLayerValues{hl_idx};
                    hiddenTF = hiddenTFValues{tf_idx};
                    
                    % Log configuration
                    fprintf('  LR: %.3f, MC: %.2f, Layers: [%s], TF: %s\n', ...
                        lr, mc, num2str(hiddenLayers), hiddenTF);
                    fprintf(optLogFid, '\nConfig %d - LR: %.3f, MC: %.2f, Layers: [%s], TF: %s\n', ...
                        configCounter, lr, mc, num2str(hiddenLayers), hiddenTF);
                    
                    % Create network settings (simplified)
                    nnSettings = struct();
                    nnSettings.hiddenLayers = hiddenLayers;
                    nnSettings.transferFunctions = repmat({hiddenTF}, 1, length(hiddenLayers));
                    nnSettings.transferFunctions{end+1} = 'purelin'; % Output layer
                    nnSettings.trainingFunction = 'trainlm';
                    nnSettings.learningRate = lr;
                    nnSettings.momentum = mc;
                    nnSettings.maxEpochs = 50; % Low number of epochs for debug mode
                    
                    % Split data - simple 80/20 split for debug
                    rng(42); % For reproducibility
                    trainIdx = randperm(size(input, 2), round(0.8 * size(input, 2)));
                    testIdx = setdiff(1:size(input, 2), trainIdx);
                    
                    trainInput = input(:, trainIdx);
                    trainTarget = target(:, trainIdx);
                    testInput = input(:, testIdx);
                    testTarget = target(:, testIdx);
                    
                    try
                        % In debug mode, just simulate training with random performance
                        fprintf('  Simulating training (debug mode)...\n');
                        
                        % Simulate network creation
                        net = struct();
                        net.layers = cell(1, length(hiddenLayers) + 1);
                        for i = 1:length(hiddenLayers)
                            net.layers{i} = struct('size', hiddenLayers(i));
                        end
                        net.layers{end} = struct('size', size(target, 1));
                        
                        % Simulate training metrics
                        trainMSE = 0.1 + rand()*0.1 - lr/10 - mc/10; % Better performance with higher lr/mc
                        valMSE = trainMSE * (1.1 + rand()*0.2);
                        testMSE = valMSE * (1.0 + rand()*0.15);
                        
                        fprintf('  Results - Train MSE: %.4f, Val MSE: %.4f, Test MSE: %.4f\n', ...
                            trainMSE, valMSE, testMSE);
                        fprintf(optLogFid, '  Train MSE: %.4f, Val MSE: %.4f, Test MSE: %.4f\n', ...
                            trainMSE, valMSE, testMSE);
                        
                        % Check if this is the best model so far
                        if valMSE < bestMSE
                            bestMSE = valMSE;
                            bestConfig = nnSettings;
                            bestConfig.trainMSE = trainMSE;
                            bestConfig.valMSE = valMSE;
                            bestConfig.testMSE = testMSE;
                            
                            % Save the "best" model
                            fprintf('  New best model found! Saving...\n');
                            fprintf(optLogFid, '  NEW BEST MODEL!\n');
                            save(bestNetFile, 'net', 'bestConfig', 'nnSettings', 'PS', 'TS');
                        end
                    catch me
                        fprintf('  Error in configuration %d: %s\n', configCounter, me.message);
                        fprintf(optLogFid, '  ERROR: %s\n', me.message);
                    end
                end
            end
        end
    end
    
    % Report best configuration
    fprintf('\nBest configuration found:\n');
    fprintf('  LR: %.3f, MC: %.2f, Layers: [%s], TF: %s\n', ...
        bestConfig.learningRate, bestConfig.momentum, ...
        num2str(bestConfig.hiddenLayers), bestConfig.transferFunctions{1});
    fprintf('  MSE - Train: %.4f, Val: %.4f, Test: %.4f\n', ...
        bestConfig.trainMSE, bestConfig.valMSE, bestConfig.testMSE);
    
    % Close log file
    fclose(optLogFid);
    fprintf('Optimization log saved to: %s\n', optLogFile);
    fprintf('Best model saved to: %s\n', bestNetFile);
    
    % Execute SHAP Analysis
    fprintf('\n===== RUNNING SIMPLIFIED SHAP ANALYSIS (DEBUG MODE) =====\n');
    fprintf('Generating simplified SHAP values...\n');

    % Load results if they exist, otherwise generate them
    results_file = fullfile(trainingDir, 'Results_trained.mat');
    if exist(results_file, 'file')
        fprintf('Loading test model data from: %s\n', results_file);
        load(results_file);
    else
        fprintf('No existing model file found. Creating simplified test model...\n');
        % Create simplified test model
        X = rand(20, 5); % 20 samples, 5 features
        Y = rand(20, 2); % 20 samples, 2 targets
        
        % Create more model-like structure
        net = struct();
        net.layers = [10, 2];
        net.transferFcn = 'tansig';
        net.performFcn = 'mse';
        net.trainFcn = 'trainbr'; % Bayesian regularization
        net.lr = 0.1;
        net.mc = 0.9;
        
        % Create simple predictions
        predictions = [0.324524756328847 0.621467592389274;
                      0.485384043745262 0.785028275287541]; % Two target values for demo
                      
        % Save to mat file for later use
        save(results_file, 'X', 'Y', 'net', 'predictions');
        fprintf('Created test model data and saved to: %s\n', results_file);
    end

    % Try to read feature names from an Excel file
    try
        raw_data_file = fullfile(rootDir, 'data', 'raw', 'RawInputData.xlsx');
        if exist(raw_data_file, 'file')
            fprintf('Trying to read real feature names from %s...\n', raw_data_file);
            [~, ~, raw] = xlsread(raw_data_file, 1);
            if size(raw, 2) >= 5
                % Grab column headers as feature names
                varNames = raw(1, 1:5);
                fprintf('Successfully read %d real feature names\n', length(varNames));
            else
                varNames = {'Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5'};
                fprintf('Excel file does not have enough columns. Using default feature names.\n');
            end
        else
            varNames = {'Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5'};
            fprintf('Raw data file not found. Using default feature names.\n');
        end
    catch
        varNames = {'Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5'};
        fprintf('Error reading feature names. Using default feature names.\n');
    end

    % Define target names
    targetNames = {'Target 1', 'Target 2'};

    % Generate simple SHAP values
    if ~exist('X', 'var')
        % If we loaded a file that doesn't have X defined
        X = rand(20, 5); % 20 samples, 5 features
        Y = rand(20, 2); % 20 samples, 2 targets
    end

    % Generate simplified SHAP values
    numSamples = size(X, 1);
    numFeatures = size(X, 2);
    numTargets = 2; % For demo

    % Generate simple SHAP values (random for demonstration)
    shapValues = zeros(numSamples, numFeatures, numTargets);
    for i = 1:numTargets
        for j = 1:numSamples
            % Make sure there's some pattern - higher feature values generally 
            % have higher SHAP contribution
            shapValues(j, :, i) = X(j, :) .* randn(1, numFeatures) * 0.1;
        end
    end

    % Processing loop for demo purposes
    for i = 1:4
        fprintf('Processed %d/%d samples\n', i*5, numSamples);
    end

    % Create base value (mean prediction)
    baseValue = [0.324524756328847 0.621467592389274];

    % Prepare for visualization
    input = X;
    featureNames = varNames;

    % Set up the shapDir for proper output
    shapDir = fullfile(rootDir, 'results', 'analysis', 'debug');

    % Create directories if they don't exist
    if ~exist(fullfile(shapDir, 'data'), 'dir')
        mkdir(fullfile(shapDir, 'data'));
    end
    if ~exist(fullfile(shapDir, 'figures'), 'dir')
        mkdir(fullfile(shapDir, 'figures'));
    end

    % Save SHAP results to data directory
    shap_results_file = fullfile(shapDir, 'data', 'shap_results.mat');
    fprintf('Saving SHAP results to: %s\n', shap_results_file);
    
    % Create a comprehensive structure with all necessary variables
    shap_data = struct();
    shap_data.shapValues = shapValues;
    shap_data.baseValue = baseValue;
    shap_data.input = input;
    shap_data.varNames = varNames;
    shap_data.featureNames = featureNames;
    shap_data.targetNames = targetNames;
    shap_data.analysisMode = 'debug';
    shap_data.creationDate = datestr(now);
    
    % Save the structure to make loading easier later
    save(shap_results_file, '-struct', 'shap_data');
    
    % Also save individual variables for backward compatibility
    save(shap_results_file, 'shapValues', 'baseValue', 'input', 'varNames', 'targetNames', 'featureNames', '-append');
    fprintf('SHAP results saved to: %s\n', shap_results_file);

    % Use the improved SHAP analysis master script to handle all visualizations and exports
    fprintf('\n===== Running Full SHAP Analysis Pipeline =====\n');
    try
        % Run the full SHAP analysis pipeline
        analysisMode = 'debug';
        parentScript = 'debug_run_analysis';  % Flag to indicate this is called from another script
        run_shap_analysis;
        fprintf('\n===== Full SHAP Analysis Pipeline Completed Successfully =====\n');
    catch ME
        fprintf('\n===== Error in Full SHAP Analysis Pipeline =====\n');
        fprintf('Error: %s\n', ME.message);
        fprintf('Stack trace:\n%s\n', getReport(ME));
        
        % Try running individual visualization scripts
        fprintf('\n===== Attempting Individual Steps =====\n');
        
        % Try visualization scripts one by one
        try
            fprintf('Creating visualizations...\n');
            plot_shap_beeswarm;
            fprintf('Beeswarm plots created successfully.\n');
        catch ME
            fprintf('Error creating beeswarm plots: %s\n', ME.message);
        end
        
        try
            plot_shap_original_summary;
            fprintf('Original summary plots created successfully.\n');
        catch ME
            fprintf('Error creating original summary plots: %s\n', ME.message);
        end
        
        try
            plot_shap_results;
            fprintf('Enhanced result plots created successfully.\n');
        catch ME
            fprintf('Error creating enhanced result plots: %s\n', ME.message);
        end
        
        % Try Excel export separately
        try
            fprintf('Exporting to Excel...\n');
            export_shap_to_excel;
            fprintf('Excel export completed successfully.\n');
        catch ME
            fprintf('Error exporting to Excel: %s\n', ME.message);
        end
    end

    fprintf('\n===== DEBUG RUN ANALYSIS COMPLETE =====\n');
    fprintf('Saved SHAP results to %s\n', shap_results_file);
catch e
    % Handle errors
    fprintf('\n===== ERROR OCCURRED =====\n');
    fprintf('Error: %s\n', e.message);
    fprintf('Stack trace:\n');
    disp(e.stack);
end

% Close diary
diary off; 