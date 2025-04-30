%% Optimized Model Training and SHAP Analysis Script
% This script handles both TVT and TT strategies for neural network training,
% addressing matrix dimension mismatch issues in TT strategy.
% It performs both model training and SHAP analysis in one script.

% Store command-line variables that might be passed from shell script
if exist('optimDir', 'var')
    stored_optimDir = optimDir;
end
if exist('bestModelDir', 'var')
    stored_bestModelDir = bestModelDir;
end
if exist('trainingDir', 'var')
    stored_trainingDir = trainingDir;
end
if exist('analysisDir', 'var')
    stored_analysisDir = analysisDir;
end
if exist('skipTraining', 'var')
    stored_skipTraining = skipTraining;
end
if exist('skipSHAP', 'var')
    stored_skipSHAP = skipSHAP;
end

% Clear workspace and command window
clear all; close all; clc;

% Start timer
startTime = tic;

% Setup diary for logging
diary('run_optimized_model_log.txt');
diary on;

fprintf('Starting run_optimized_model.m\n');
fprintf('======================================\n\n');

% Restore variables from command line if they were saved
if exist('stored_optimDir', 'var')
    optimizationDir = stored_optimDir; % Note variable name change to match script
    fprintf('Using optimizationDir from command line: %s\n', optimizationDir);
    clear stored_optimDir;
end
if exist('stored_bestModelDir', 'var')
    bestModelDir = stored_bestModelDir;
    fprintf('Using bestModelDir from command line: %s\n', bestModelDir);
    clear stored_bestModelDir;
end
if exist('stored_trainingDir', 'var')
    trainingDir = stored_trainingDir;
    fprintf('Using trainingDir from command line: %s\n', trainingDir);
    clear stored_trainingDir;
end
if exist('stored_analysisDir', 'var')
    analysisDir = stored_analysisDir;
    fprintf('Using analysisDir from command line: %s\n', analysisDir);
    clear stored_analysisDir;
end
if exist('stored_skipTraining', 'var')
    skipTraining = stored_skipTraining;
    fprintf('Using skipTraining from command line: %s\n', skipTraining);
    clear stored_skipTraining;
else
    skipTraining = false;  % Default: Don't skip training
end
if exist('stored_skipSHAP', 'var')
    skipSHAP = stored_skipSHAP;
    fprintf('Using skipSHAP from command line: %s\n', skipSHAP);
    clear stored_skipSHAP;
else
    skipSHAP = false;  % Default: Don't skip SHAP analysis
end

try
    %% Setup directories and paths
    % Get the root directory
    scriptPath = mfilename('fullpath');
    [scriptDir, ~, ~] = fileparts(scriptPath);
    rootDir = fileparts(fileparts(scriptDir));
    fprintf('Root directory: %s\n', rootDir);
    
    % Set paths to various directories if not already defined
    if ~exist('modelDir', 'var')
        modelDir = fullfile(rootDir, 'src', 'model');
    end
    
    if ~exist('resultsDir', 'var')
        resultsDir = fullfile(rootDir, 'results');
    end
    
    if ~exist('bestModelDir', 'var')
        bestModelDir = fullfile(resultsDir, 'best_model');
    end
    
    if ~exist('trainingDir', 'var')
        trainingDir = fullfile(resultsDir, 'training');
    end
    
    if ~exist('optimizationDir', 'var')
        optimizationDir = fullfile(resultsDir, 'optimization');
    end
    
    if ~exist('analysisDir', 'var')
        analysisDir = fullfile(resultsDir, 'analysis', 'full');
    end
    
    % Create directories if they don't exist
    directories = {resultsDir, bestModelDir, trainingDir, optimizationDir, ...
                  analysisDir, fullfile(analysisDir, 'data'), fullfile(analysisDir, 'figures'), ...
                  fullfile(bestModelDir, 'Figures')};
    
    for i = 1:length(directories)
        if ~exist(directories{i}, 'dir')
            mkdir(directories{i});
            fprintf('Created directory: %s\n', directories{i});
        end
    end
    
    % Add necessary paths
    addpath(modelDir);
    addpath(scriptDir);
    addpath(fullfile(rootDir, 'src', 'visualization'));
    
    % Check for required functions
    requiredFunctions = {'nncreate', 'nnpredict', 'nnpreprocess', 'nntrain', 'nneval', 'nnff'};
    missingFunctions = false;
    for i = 1:length(requiredFunctions)
        if ~exist(fullfile(modelDir, [requiredFunctions{i} '.m']), 'file')
            fprintf('WARNING: Required function not found: %s\n', ...
                fullfile(modelDir, [requiredFunctions{i} '.m']));
            missingFunctions = true;
        end
    end
    
    if missingFunctions
        error('Missing required model functions. Cannot proceed.');
    end
    
    %% Load and validate optimization results
    fprintf('\nLoading best hyperparameters from optimization...\n');
    
    % Load the best configuration from optimization results
    optimResultsFile = fullfile(optimizationDir, 'best_model.mat');
    if ~exist(optimResultsFile, 'file')
        error('ERROR: Best model optimization results not found at %s', optimResultsFile);
    end
    
    % Load optimization results
    optimResults = load(optimResultsFile);
    if ~isfield(optimResults, 'best_config')
        error('ERROR: Invalid optimization results file. Missing best_config field.');
    end
    
    best_config = optimResults.best_config;
    fprintf('Best configuration loaded successfully\n');
    
    % Display the best configuration
    configFields = fieldnames(best_config);
    for i = 1:length(configFields)
        if isnumeric(best_config.(configFields{i}))
            fprintf('%s: %g\n', configFields{i}, best_config.(configFields{i}));
        elseif ischar(best_config.(configFields{i}))
            fprintf('%s: %s\n', configFields{i}, best_config.(configFields{i}));
        elseif iscell(best_config.(configFields{i}))
            fprintf('%s: %s\n', configFields{i}, mat2str(best_config.(configFields{i})));
        end
    end
    
    %% Determine training strategy and parameters
    fprintf('\nDetermining training strategy and parameters...\n');
    
    % Extract strategy from best_config
    if isfield(best_config, 'strategy')
        strategy = best_config.strategy;
    else
        strategy = 'TT'; % Default to Train-Test strategy
        fprintf('Strategy not found in best_config, defaulting to TT\n');
    end
    
    fprintf('Using training strategy: %s\n', strategy);
    
    % Extract training parameters
    if isfield(best_config, 'trainRatio')
        trainRatio = best_config.trainRatio;
    else
        trainRatio = 0.7; % Default
        fprintf('trainRatio not found in best_config, defaulting to 0.7\n');
    end
    
    if isfield(best_config, 'valRatio')
        valRatio = best_config.valRatio;
    else
        if strcmp(strategy, 'TVT')
            valRatio = 0.15; % Default for TVT
            fprintf('valRatio not found in best_config, defaulting to 0.15 for TVT\n');
        else
            valRatio = 0; % No validation set for TT
            fprintf('Setting valRatio to 0 for TT strategy\n');
        end
    end
    
    % Force valRatio to 0 for TT strategy to prevent issues
    if strcmp(strategy, 'TT') && valRatio > 0
        fprintf('WARNING: Strategy is TT but valRatio > 0. Setting valRatio to 0.\n');
        valRatio = 0;
    end
    
    % Extract other model parameters
    if isfield(best_config, 'hiddenLayer')
        hiddenLayer = best_config.hiddenLayer;
    else
        hiddenLayer = [10]; % Default
        fprintf('hiddenLayer not found in best_config, defaulting to [10]\n');
    end
    
    if isfield(best_config, 'hiddenTF')
        hiddenTF = best_config.hiddenTF;
    else
        hiddenTF = 'tansig'; % Default
        fprintf('hiddenTF not found in best_config, defaulting to tansig\n');
    end
    
    if isfield(best_config, 'outputTF')
        outputTF = best_config.outputTF;
    else
        outputTF = 'purelin'; % Default
        fprintf('outputTF not found in best_config, defaulting to purelin\n');
    end
    
    % Learning parameters
    if isfield(best_config, 'lr')
        lr = best_config.lr;
    else
        lr = 0.01; % Default
        fprintf('lr not found in best_config, defaulting to 0.01\n');
    end
    
    if isfield(best_config, 'mc')
        mc = best_config.mc;
    else
        mc = 0.7; % Default
        fprintf('mc not found in best_config, defaulting to 0.7\n');
    end
    
    if isfield(best_config, 'lr_inc')
        lr_inc = best_config.lr_inc;
    else
        lr_inc = 1.05; % Default
        fprintf('lr_inc not found in best_config, defaulting to 1.05\n');
    end
    
    if isfield(best_config, 'lr_dec')
        lr_dec = best_config.lr_dec;
    else
        lr_dec = 0.7; % Default
        fprintf('lr_dec not found in best_config, defaulting to 0.7\n');
    end
    
    % Set epochs for training
    epochs = 6000; % Standard for full training
    
    %% Load and prepare data
    fprintf('\nLoading and preparing data for model training...\n');
    
    % Check and load data files
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
    
    % Load RawInputData.xlsx to determine feature and target dimensions
    try
        fprintf('Loading data from %s...\n', rawDataFile);
        [~, ~, raw] = xlsread(rawDataFile);
        [numRows, numCols] = size(raw);
        numSamples = numRows - 1; % Subtract 1 for header row
        
        % Based on bpDNN4PyroProd.m, target variables start at column 14
        inputFeatureEndCol = 13; % The last column index for input features before adding feedstock data
        targetStartCol = 14;     % The first column index where target variables begin
        numTargetCols = numCols - targetStartCol + 1; % Calculate how many target columns exist
        
        fprintf('Sample count determined: %d samples\n', numSamples);
        fprintf('Detected %d target variable columns\n', numTargetCols);
    catch readErr
        error('Error reading RawInputData.xlsx: %s', readErr.message);
    end
    
    % Load CopyrolysisFeedstock.mat to determine feature count
    try
        feedstockData = load(feedstockFile);
        if isfield(feedstockData, 'CopyrolysisFeedstockTag')
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
    
    % Step 1: Get prepared data from bpDNN4PyroProd
    try
        fprintf('Getting prepared data from bpDNN4PyroProd...\n');
        addpath(modelDir);
        [input_data, target_data, PS_global, TS_global] = bpDNN4PyroProd();
        fprintf('Data preparation complete - got %d input features and %d targets\n', size(input_data, 1), size(target_data, 1));
    catch ME
        error('Error in bpDNN4PyroProd data preparation: %s', ME.message);
    end
    
    %% Configure and train neural network
    if ~skipTraining
        fprintf('Starting model training step...\n');
        try
            fprintf('\nConfiguring neural network with best parameters...\n');
            fprintf('Converting scalar hiddenLayer to array: %s\n', mat2str(hiddenLayer));
            
            % Create new neural network
            fprintf('Creating neural network with structure: %s\n', mat2str(hiddenLayer));
            
            % Pre-process data
            [PS, input, TS, target] = nnpreprocess('mapminmax', input_data, target_data);
            numInputs = size(input, 1);
            numOutputs = size(target, 1);
            
            % Create network with correct architecture
            net = nncreate(numInputs, hiddenLayer, numOutputs, hiddenTF, outputTF);
            
            % Set network parameters based on training strategy
            net.divideFcn = 'dividerand';
            net.divideParam.trainRatio = trainRatio;
            net.divideParam.valRatio = valRatio;
            net.divideParam.testRatio = 1 - trainRatio - valRatio;
            
            % Set training parameters
            net.trainParam.goal = 1e-5;
            net.trainParam.epoch = epochs;
            net.trainParam.min_grad = 1e-6;
            net.trainParam.max_fail = 10;
            net.trainParam.lr = lr;
            net.trainParam.mc = mc;
            net.trainParam.lr_inc = lr_inc;
            net.trainParam.lr_dec = lr_dec;
            net.trainParam.showWindow = true;
            net.trainParam.showCommandLine = true;
            
            fprintf('Using %s training strategy\n', strategy);
            fprintf('Data division ratios: %.2f training, %.2f validation, %.2f testing\n', ...
                trainRatio, valRatio, net.divideParam.testRatio);
            fprintf('Epochs: %d, Learning rate: %.4f, Momentum: %.2f\n', ...
                epochs, lr, mc);
            
            % Train the network with appropriate error handling
            fprintf('\nTraining neural network...\n');
            fprintf('===== STARTING NETWORK TRAINING =====\n');
            [net, tr] = nntrain(net, input, target);
            fprintf('Network training completed successfully\n');
            
            % Extract performance metrics
            trainPerf = tr.perf;
            bestEpoch = tr.best_epoch;
            finalTrainPerf = trainPerf(end);
            bestTrainPerf = tr.best_perf;
            
            fprintf('Training completed in %d epochs\n', tr.num_epochs);
            fprintf('Best epoch: %d\n', bestEpoch);
            fprintf('Final training MSE: %.6f\n', finalTrainPerf);
            fprintf('Best training MSE: %.6f\n', bestTrainPerf);
            
            if isfield(tr, 'best_vperf') && ~isempty(tr.best_vperf)
                fprintf('Best validation MSE: %.6f\n', tr.best_vperf);
            end
            
            if isfield(tr, 'best_tperf') && ~isempty(tr.best_tperf)
                fprintf('Best test MSE: %.6f\n', tr.best_tperf);
            end
            
            % Training succeeded - save the model
            fprintf('\nSaving best model...\n');
            save(fullfile(bestModelDir, 'best_model.mat'), 'net', 'input_data', 'target_data', ...
                'PS', 'TS', 'PS_global', 'TS_global', 'strategy', 'trainRatio', 'valRatio', ...
                'lr', 'mc', 'lr_inc', 'lr_dec', 'hiddenLayer', 'hiddenTF', 'outputTF');
            
            % Save a copy to training directory for SHAP analysis
            save(fullfile(trainingDir, 'trained_model.mat'), 'net', 'input_data', 'target_data', ...
                'PS', 'TS', 'PS_global', 'TS_global', 'strategy', 'trainRatio', 'valRatio', ...
                'lr', 'mc', 'lr_inc', 'lr_dec', 'hiddenLayer', 'hiddenTF', 'outputTF');
            
            fprintf('Model saved to:\n  %s\n  %s\n', ...
                fullfile(bestModelDir, 'best_model.mat'), ...
                fullfile(trainingDir, 'trained_model.mat'));
            
        catch ME
            error('Error training neural network: %s', ME.message);
        end
        fprintf('Model training step completed successfully.\n');
    else
        fprintf('Skipping model training step as requested (skipTraining=true).\n');
        fprintf('Will use existing trained model for SHAP analysis.\n');
        
        % Verify the trained model file exists
        if ~exist(fullfile(bestModelDir, 'best_model.mat'), 'file')
            error('Cannot skip training: Trained model file not found at %s', fullfile(bestModelDir, 'best_model.mat'));
        end
        
        % Load the existing model
        fprintf('Loading existing trained model for SHAP analysis...\n');
        try
            modelData = load(fullfile(bestModelDir, 'best_model.mat'));
            if ~isfield(modelData, 'net')
                error('Invalid trained model: Missing neural network model');
            end
            
            % Extract the trained network and data
            net = modelData.net;
            if isfield(modelData, 'tr')
                tr = modelData.tr;
            end
            
            % Load other required variables if they exist
            if isfield(modelData, 'input_data')
                input_data = modelData.input_data;
            end
            if isfield(modelData, 'target_data')
                target_data = modelData.target_data;
            end
            if isfield(modelData, 'PS')
                PS = modelData.PS;
            end
            if isfield(modelData, 'TS')
                TS = modelData.TS;
            end
            
            fprintf('Loaded existing trained model successfully.\n');
        catch ME
            error('Error loading existing trained model: %s', ME.message);
        end
    end
    
    %% Perform SHAP Analysis
    if ~skipSHAP
        try
            fprintf('Starting SHAP analysis step...\n');
            fprintf('\n===== STARTING SHAP ANALYSIS =====\n');
            
            % Set up SHAP directory and parameters
            shapDir = analysisDir;
            dataDir = fullfile(shapDir, 'data');
            figDir = fullfile(shapDir, 'figures');
            
            % Make sure required variables are defined
            if ~exist('shapDir', 'var')
                shapDir = analysisDir;
            end
            
            % Check for the trained model before proceeding
            modelFile = fullfile(trainingDir, 'trained_model.mat');
            if ~exist(modelFile, 'file')
                modelFile = fullfile(bestModelDir, 'best_model.mat');
                if ~exist(modelFile, 'file')
                    error('No trained model found for SHAP analysis');
                end
            end
            
            fprintf('Using model file: %s\n', modelFile);
            
            % Load the model
            try
                modelData = load(modelFile);
                
                % Extract data components from loaded model
                if isfield(modelData, 'input_data')
                    input = modelData.input_data;
                elseif isfield(modelData, 'X')
                    input = modelData.X;
                else
                    error('No input data found in model file');
                end
                
                if isfield(modelData, 'target_data')
                    target = modelData.target_data;
                elseif isfield(modelData, 'Y')
                    target = modelData.Y;
                else
                    error('No target data found in model file');
                end
                
                if isfield(modelData, 'net')
                    net = modelData.net;
                else
                    error('No neural network model found in model file');
                end
                
                fprintf('Model loaded successfully for SHAP analysis\n');
            catch loadErr
                error('Error loading model for SHAP analysis: %s', loadErr.message);
            end
            
            % Call calc_shap_values.m to compute SHAP values
            fprintf('Calculating SHAP values...\n');
            
            % Check if calc_shap_values.m exists
            calc_shap_script = fullfile(scriptDir, 'calc_shap_values.m');
            if ~exist(calc_shap_script, 'file')
                error('calc_shap_values.m not found at %s', calc_shap_script);
            end
            
            % Set up variables needed by calc_shap_values.m
            useAllSamples = true;  % Use all available samples
            useAllFeatures = true; % Use all features
            
            % Run the SHAP calculation script
            try
                run(calc_shap_script);
                
                % Check if SHAP values were generated
                resultFile = fullfile(dataDir, 'shap_results.mat');
                if ~exist(resultFile, 'file')
                    error('calc_shap_values.m did not generate shap_results.mat');
                else
                    fprintf('SHAP calculation completed successfully.\n');
                    fprintf('SHAP results saved to: %s\n', resultFile);
                end
            catch shapErr
                fprintf('Error in SHAP calculation: %s\n', shapErr.message);
                fprintf('SHAP analysis failed. Check error message for details.\n');
            end
            
            % Export SHAP results to Excel
            fprintf('\nExporting SHAP values to Excel...\n');
            
            % Check if export_shap_to_excel.m exists
            export_script = fullfile(scriptDir, 'export_shap_to_excel.m');
            if ~exist(export_script, 'file')
                fprintf('WARNING: export_shap_to_excel.m not found at %s\n', export_script);
                fprintf('Skipping Excel export step.\n');
            else
                try
                    % Run the Excel export script
                    run(export_script);
                    fprintf('SHAP results exported to Excel successfully.\n');
                catch exportErr
                    fprintf('Error in Excel export: %s\n', exportErr.message);
                    fprintf('Excel export failed. Check error message for details.\n');
                end
            end
            
            % Generate SHAP visualizations
            fprintf('\nGenerating SHAP visualizations...\n');
            
            % Check if plot_shap_results.m exists
            plot_script = fullfile(rootDir, 'src', 'visualization', 'plot_shap_results.m');
            if ~exist(plot_script, 'file')
                fprintf('WARNING: plot_shap_results.m not found at %s\n', plot_script);
                fprintf('Skipping SHAP visualization step.\n');
            else
                try
                    % Run the SHAP visualization script
                    run(plot_script);
                    fprintf('SHAP visualizations generated successfully.\n');
                catch plotErr
                    fprintf('Error in SHAP visualization: %s\n', plotErr.message);
                    fprintf('Visualization failed. Check error message for details.\n');
                end
            end
            
            fprintf('SHAP analysis step completed.\n');
        catch ME
            fprintf('WARNING: Error in SHAP analysis: %s\n', ME.message);
            fprintf('Continuing with workflow despite SHAP analysis error\n');
        end
    else
        fprintf('Skipping SHAP analysis step as requested (skipSHAP=true).\n');
        fprintf('Only model training will be performed.\n');
    end
    
    %% Final timing and summary
    elapsedTime = toc(startTime);
    fprintf('\n======================================\n');
    fprintf('Script completed in %.2f seconds (%.2f minutes)\n', ...
        elapsedTime, elapsedTime/60);
    
    % Summary of results
    fprintf('\nExecution Summary:\n');
    
    % Check if training was successful
    trainingSuccessful = exist('tr', 'var');
    if trainingSuccessful
        fprintf('- Model training: Successful\n');
        fprintf('  - Training epochs: %d\n', tr.num_epochs);
        fprintf('  - Best epoch: %d\n', tr.best_epoch);
        fprintf('  - Final MSE: %.6f\n', tr.perf(end));
        fprintf('  - Best MSE: %.6f\n', tr.best_perf);
    else
        fprintf('- Model training: Failed\n');
    end
    
    % Check if SHAP analysis was successful
    shapAnalysisCompleted = exist(fullfile(dataDir, 'shap_results.mat'), 'file');
    if shapAnalysisCompleted
        fprintf('- SHAP analysis: Completed\n');
    else
        fprintf('- SHAP analysis: Failed\n');
    end
    
    % Check if Excel export was successful
    excelExportCompleted = exist(fullfile(shapDir, 'shap_analysis_results.xlsx'), 'file');
    if excelExportCompleted
        fprintf('- Excel export: Completed\n');
    else
        fprintf('- Excel export: Not completed\n');
    end
    
    % List output files
    fprintf('\nOutput Files:\n');
    fprintf('- Best model: %s\n', fullfile(bestModelDir, 'best_model.mat'));
    fprintf('- Training model: %s\n', fullfile(trainingDir, 'trained_model.mat'));
    fprintf('- SHAP results: %s\n', fullfile(dataDir, 'shap_results.mat'));
    
    if exist(fullfile(shapDir, 'shap_analysis_results.xlsx'), 'file')
        fprintf('- Excel analysis: %s\n', fullfile(shapDir, 'shap_analysis_results.xlsx'));
    end
    
    % Provide next steps
    fprintf('\nNext Steps:\n');
    fprintf('- Review the trained model and performance metrics\n');
    fprintf('- Examine the SHAP analysis results and visualizations\n');
    fprintf('- Check the Excel exports for detailed feature importance data\n');
    
    fprintf('\nrun_optimized_model.m completed successfully\n');
    
catch mainErr
    % Handle overall script errors
    fprintf('\n===== ERROR IN MAIN SCRIPT =====\n');
    fprintf('Error message: %s\n', mainErr.message);
    fprintf('Error in: %s (Line %d)\n', mainErr.stack(1).name, mainErr.stack(1).line);
    
    % Get more detailed error info
    errorDetails = getReport(mainErr, 'extended');
    fprintf('\nDetailed Error Report:\n%s\n', errorDetails);
    
    fprintf('\nrun_optimized_model.m completed with errors\n');
    
    % Rethrow the error to ensure it propagates to the MATLAB command line
    rethrow(mainErr);
end

% End diary logging
diary off;

% Print final message to console only (not logged)
fprintf('\nScript execution completed. See run_optimized_model_log.txt for detailed log.\n'); 