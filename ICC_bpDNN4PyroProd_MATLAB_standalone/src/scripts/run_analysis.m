%% Simplified Analysis Script for Neural Network Testing
% This script performs simplified testing of the neural network model

% Clear workspace and command window
%clc; 
%clear all; 
%close all;

% Start timing execution
analysisStartTime = tic;

% Setup logging to file
logFile = fullfile(pwd, 'run_analysis_log.txt');
diary(logFile);
fprintf('Logging to file: %s\n', logFile);

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
    modelDir = fullfile(rootDir, 'src', 'model'); % Using src/model instead of model_data
    tempTrainingDir = fullfile(rootDir, 'results', 'training'); % Using results/training instead of training_data
    bestModelDir = fullfile(rootDir, 'results', 'best_model'); % Best model directory
    optimizationDir = fullfile(rootDir, 'results', 'optimization'); % Optimization directory
    
    % Check if analysisMode is defined, default to 'debug' if not
    if ~exist('analysisMode', 'var')
        analysisMode = 'debug';
        fprintf('analysisMode not defined, defaulting to debug mode\n');
    else
        fprintf('Running in %s mode\n', analysisMode);
    end
    
    % Set up analysis directory based on mode
    if strcmpi(analysisMode, 'full')
        analysisDir = fullfile(rootDir, 'results', 'analysis', 'full');
        fprintf('Using full analysis directory: %s\n', analysisDir);
    else
        analysisDir = fullfile(rootDir, 'results', 'analysis', 'debug');
        fprintf('Using debug analysis directory: %s\n', analysisDir);
    end
    
    % Make sure shapDir points to the correct analysis directory
    shapDir = analysisDir;
    fprintf('Setting shapDir to %s\n', shapDir);
    
    % Create the data directories
    analysisDataDir = fullfile(analysisDir, 'data');
    analysisFigDir = fullfile(analysisDir, 'figures');
    
    % Check if directories exist, create if they don't
    dirs_to_check = {};
    dirs_to_check{1} = tempTrainingDir;
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
    
    % Check for and load existing model file based on analysis mode
    if strcmpi(analysisMode, 'full')
        % In full mode, we should use the best model from training
        bestModelFile = fullfile(bestModelDir, 'best_model.mat');
        trainedModelFile = fullfile(tempTrainingDir, 'Results_trained.mat');
        
        if exist(bestModelFile, 'file')
            fprintf('Loading best model from: %s\n', bestModelFile);
            modelData = load(bestModelFile);
            modelFileUsed = bestModelFile;
        elseif exist(trainedModelFile, 'file')
            fprintf('Best model not found, loading trained model from: %s\n', trainedModelFile);
            modelData = load(trainedModelFile);
            modelFileUsed = trainedModelFile;
        else
            error('ERROR: No trained model found. Cannot proceed with analysis.');
        end
    else
        % Even in debug mode, we still need a trained model
        trainedModelFile = fullfile(tempTrainingDir, 'Results_trained.mat');
        if exist(trainedModelFile, 'file')
            fprintf('Loading trained model from: %s\n', trainedModelFile);
            modelData = load(trainedModelFile);
            modelFileUsed = trainedModelFile;
        else
            error('ERROR: No trained model found. Cannot proceed with analysis.');
        end
    end
    
    % Extract data components from loaded model
    if isfield(modelData, 'input')
        input = modelData.input;
    elseif isfield(modelData, 'X')
        input = modelData.X;
        fprintf('Using X as input\n');
    else
        error('No input data found in model file');
    end
    
    if isfield(modelData, 'target')
        target = modelData.target;
    elseif isfield(modelData, 'Y')
        target = modelData.Y;
        fprintf('Using Y as target\n');
    else
        error('No target data found in model file');
    end
    
    if isfield(modelData, 'net')
        net = modelData.net;
    else
        error('No neural network model found in model file');
    end
    
    % Add model directory to path if it's not already
    if ~contains(path, modelDir)
        addpath(modelDir);
        fprintf('Added model directory to path: %s\n', modelDir);
    end
    
    % Run the SHAP analysis based on mode
    if strcmpi(analysisMode, 'full')
        fprintf('\n===== RUNNING FULL SHAP ANALYSIS =====\n');
        
        % Call calc_shap_values.m to compute real SHAP values
        fprintf('Calculating SHAP values using calc_shap_values.m...\n');
        
        % Make sure required variables are defined for calc_shap_values.m
        if ~exist('shapDir', 'var')
            shapDir = analysisDir;
        end
        
        % Check if calc_shap_values.m exists and call it
        calc_shap_script = fullfile(scriptDir, 'calc_shap_values.m');
        if exist(calc_shap_script, 'file')
            % Set up any additional variables needed by calc_shap_values.m
            featureNames = {}; % Will be created by calc_shap_values.m if not available
            
            % Run the SHAP calculation script
            run(calc_shap_script);
            
            % Check if SHAP values were generated
            resultFile = fullfile(analysisDataDir, 'shap_results.mat');
            if ~exist(resultFile, 'file')
                error('calc_shap_values.m did not generate shap_results.mat');
            else
                fprintf('SHAP calculation completed successfully.\n');
                fprintf('SHAP results saved to: %s\n', resultFile);
            end
        else
            error('calc_shap_values.m not found at %s', calc_shap_script);
        end
    else
        fprintf('\n===== RUNNING DEBUG MODE SHAP ANALYSIS =====\n');
        
        % In debug mode, we still use the calc_shap_values.m script, just with smaller sample sizes
        fprintf('Calculating SHAP values using calc_shap_values.m (debug mode)...\n');
        
        % Make sure required variables are defined for calc_shap_values.m
        if ~exist('shapDir', 'var')
            shapDir = analysisDir;
        end
        
        % Set debug parameters
        useAllSamples = false;
        numShapSamples = min(5, size(input, 2)); % Limit to 5 samples for debug
        fprintf('Debug mode: Using limited sample count (%d) for faster execution\n', numShapSamples);
        
        % Check if calc_shap_values.m exists and call it
        calc_shap_script = fullfile(scriptDir, 'calc_shap_values.m');
        if exist(calc_shap_script, 'file')
            % Set up any additional variables needed by calc_shap_values.m
            featureNames = {}; % Will be created by calc_shap_values.m if not available
            
            % Run the SHAP calculation script
            run(calc_shap_script);
            
            % Check if SHAP values were generated
            resultFile = fullfile(analysisDataDir, 'shap_results.mat');
            if ~exist(resultFile, 'file')
                error('calc_shap_values.m did not generate shap_results.mat');
            else
                fprintf('SHAP calculation completed successfully in debug mode.\n');
                fprintf('SHAP results saved to: %s\n', resultFile);
            end
        else
            error('calc_shap_values.m not found at %s', calc_shap_script);
        end
    end
    
    % Create visualizations if the visualization scripts exist
    fprintf('\n===== Creating Visualizations =====\n');
    
    % Check for visualization functions
    plot_shap_results_path = fullfile(rootDir, 'src', 'visualization', 'plot_shap_results.m');
    plot_shap_beeswarm_path = fullfile(rootDir, 'src', 'visualization', 'plot_shap_beeswarm.m');
    plot_shap_original_summary_path = fullfile(rootDir, 'src', 'visualization', 'plot_shap_original_summary.m');
    fix_colorbar_style_path = fullfile(rootDir, 'src', 'visualization', 'fix_colorbar_style.m');
    
    % Load SHAP results if not already in workspace
    if ~exist('shapValues', 'var') || ~exist('baseValue', 'var') || ~exist('varNames', 'var')
        resultFile = fullfile(analysisDataDir, 'shap_results.mat');
        if exist(resultFile, 'file')
            fprintf('Loading SHAP results from: %s\n', resultFile);
            results = load(resultFile);
            shapValues = results.shapValues;
            baseValue = results.baseValue;
            varNames = results.varNames;
            if isfield(results, 'input')
                input = results.input;
            end
            if isfield(results, 'target')
                target = results.target;
            end
        else
            error('SHAP results file not found and shapValues not in workspace');
        end
    end
    
    % Load variables needed for plotting
    featureNames = varNames;
    X = input; % Ensure X is defined for some visualization scripts that might need it
    y = target; % Ensure y is defined for some visualization scripts
    
    % Track visualization success
    visualization_success = false;
    
    % Create plots if visualization scripts exist
    viz_scripts = {
        {plot_shap_results_path, 'plot_shap_results.m'},
        {plot_shap_beeswarm_path, 'plot_shap_beeswarm.m'},
        {plot_shap_original_summary_path, 'plot_shap_original_summary.m'},
        {fix_colorbar_style_path, 'fix_colorbar_style.m'}
    };
    
    for i = 1:length(viz_scripts)
        script_path = viz_scripts{i}{1};
        script_name = viz_scripts{i}{2};
        
        try
            if exist(script_path, 'file')
                fprintf('Calling %s...\n', script_name);
                
                % Add more detailed verification before running script
                fprintf('  Checking requirements for %s:\n', script_name);
                fprintf('    - shapValues: %d x %d x %d array\n', size(shapValues, 1), size(shapValues, 2), size(shapValues, 3));
                fprintf('    - baseValue: %s\n', mat2str(baseValue));
                fprintf('    - varNames: %d cell array\n', length(varNames));
                fprintf('    - featureNames: %d cell array\n', length(featureNames));
                
                run(script_path);
                fprintf('  Successfully executed %s\n', script_name);
                visualization_success = true;
            else
                fprintf('Warning: %s not found at %s\n', script_name, script_path);
                
                % Try alternative paths
                alternative_paths = {
                    fullfile(rootDir, 'src', 'scripts', script_name),
                    fullfile(rootDir, 'src', 'shap', script_name)
                };
                
                for j = 1:length(alternative_paths)
                    alt_path = alternative_paths{j};
                    if exist(alt_path, 'file')
                        fprintf('  Found alternative path for %s: %s\n', script_name, alt_path);
                        try
                            fprintf('  Attempting to run from alternative path...\n');
                            run(alt_path);
                            fprintf('  Successfully executed %s from alternative path\n', script_name);
                            visualization_success = true;
                            break;
                        catch alt_err
                            fprintf('  Error running %s from alternative path: %s\n', script_name, alt_err.message);
                            if ~isempty(alt_err.stack)
                                fprintf('    Line: %d\n', alt_err.stack(1).line);
                                fprintf('    File: %s\n', alt_err.stack(1).file);
                            end
                        end
                    end
                end
            end
        catch viz_error
            fprintf('Error during %s: %s\n', script_name, viz_error.message);
            
            % Print detailed error information
            if ~isempty(viz_error.stack)
                fprintf('  Error in file: %s\n', viz_error.stack(1).file);
                fprintf('  Line: %d\n', viz_error.stack(1).line);
            end
            
            % Continue with next visualization script
            fprintf('  Continuing with other visualization scripts...\n');
        end
    end

    % Export to Excel if in full mode
    if strcmpi(analysisMode, 'full')
        fprintf('\n===== Exporting SHAP values to Excel =====\n');
        
        % Check if export_shap_to_excel.m exists and call it
        export_script = fullfile(scriptDir, 'export_shap_to_excel.m');
        if exist(export_script, 'file')
            try
                fprintf('Running export_shap_to_excel.m...\n');
                run(export_script);
                fprintf('Excel export completed successfully.\n');
            catch export_error
                fprintf('Error during Excel export: %s\n', export_error.message);
                if ~isempty(export_error.stack)
                    fprintf('  Error in file: %s\n', export_error.stack(1).file);
                    fprintf('  Line: %d\n', export_error.stack(1).line);
                end
            end
        else
            fprintf('Warning: export_shap_to_excel.m not found at %s\n', export_script);
        end
    end

    if ~visualization_success
        fprintf('\nWarning: All visualization attempts failed. SHAP values were calculated and saved,\n');
        fprintf('but no visualizations were created. Please check the visualization scripts.\n');
    end
    
    % Calculate and report total execution time
    executionTime = toc(analysisStartTime);
    fprintf('\n===== ANALYSIS EXECUTION COMPLETE =====\n');
    fprintf('Total execution time: %.2f seconds (%.2f minutes)\n', executionTime, executionTime/60);
    
    % Run integrity check on output files
    fprintf('\n===== OUTPUT FILE INTEGRITY CHECK =====\n');
    checkFiles = {
        modelFileUsed,
        resultFile,
        analysisFigDir
    };
    
    for i = 1:length(checkFiles)
        if exist(checkFiles{i}, 'file') || exist(checkFiles{i}, 'dir')
            fprintf('[OK] %s exists\n', checkFiles{i});
        else
            fprintf('[ERROR] %s does not exist\n', checkFiles{i});
        end
    end
    
    fprintf('\n\nRun analysis completed successfully!\n');
catch e
    % Handle errors
    fprintf('\n===== ERROR OCCURRED =====\n');
    fprintf('Error: %s\n', e.message);
    fprintf('Stack trace:\n');
    disp(e.stack);
end

% Close diary
diary off; 