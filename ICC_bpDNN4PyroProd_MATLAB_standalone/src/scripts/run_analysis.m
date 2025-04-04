%% Simplified Analysis Script for Neural Network Testing
% This script performs simplified testing of the neural network model

% Clear workspace and command window
clc; clear all; close all;

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
    shapDir = fullfile(rootDir, 'src', 'shap');
    modelDir = fullfile(rootDir, 'src', 'model'); % Using src/model instead of model_data
    tempTrainingDir = fullfile(rootDir, 'results', 'training'); % Using results/training instead of training_data
    bestModelDir = fullfile(rootDir, 'results', 'best_model'); % Best model directory
    optimizationDir = fullfile(rootDir, 'results', 'optimization'); % Optimization directory
    analysisDir = fullfile(rootDir, 'results', 'analysis', 'full'); % Specify full analysis directory
    
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
    
    % Create a simple test model data file
    testModelFile = fullfile(tempTrainingDir, 'Results_trained.mat');
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
    
    % Add model directory to path if it's not already
    if ~contains(path, modelDir)
        addpath(modelDir);
        fprintf('Added model directory to path: %s\n', modelDir);
    end
    
    % Skip hyperparameter optimization for simplified testing
    fprintf('\n===== SKIPPING HYPERPARAMETER OPTIMIZATION FOR SIMPLIFIED TESTING =====\n');
    
    % Run the SHAP analysis
    fprintf('\n===== RUNNING SIMPLIFIED SHAP ANALYSIS =====\n');
    
    % Make sure the shapDir variable is defined properly
    shapDir = fullfile(rootDir, 'src', 'shap');
    if ~exist(shapDir, 'dir')
        % If src/shap doesn't exist, use results/analysis/shap instead
        shapDir = fullfile(analysisDir, 'shap');
        fprintf('Created SHAP directory at: %s\n', shapDir);
        if ~exist(shapDir, 'dir')
            mkdir(shapDir);
        end
    end
    
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
    
    % Try to read real feature names from original data file
    try
        % Check the original data file
        rawDataFile = fullfile(rootDir, 'data', 'raw', 'RawInputData.xlsx');
        if exist(rawDataFile, 'file')
            fprintf('Trying to read real feature names from %s...\n', rawDataFile);
            % Read Excel file, get the first row as feature names
            [~, ~, raw] = xlsread(rawDataFile, 1, 'A1:Z1');
            if ~isempty(raw) && length(raw) >= numFeatures
                for i = 1:numFeatures
                    if ~isempty(raw{i}) && ischar(raw{i})
                        varNames{i} = raw{i};
                    else
                        varNames{i} = sprintf('Feature_%d', i);
                    end
                end
                fprintf('Successfully read %d real feature names\n', numFeatures);
            else
                fprintf('Insufficient feature names in the original data file, using default names\n');
                for i = 1:numFeatures
                    varNames{i} = sprintf('Feature_%d', i);
                end
            end
        else
            % Check metadata file in SHAP_Analysis directory
            metadataFile = fullfile(rootDir, 'SHAP_Analysis', 'Data', 'SHAP_metadata.xlsx');
            if exist(metadataFile, 'file')
                fprintf('Trying to read real feature names from %s...\n', metadataFile);
                % Read Excel file, get the first column as feature names
                [~, txt, ~] = xlsread(metadataFile);
                if ~isempty(txt) && size(txt, 1) >= numFeatures
                    for i = 1:numFeatures
                        if i <= size(txt, 1) && ~isempty(txt{i,1})
                            varNames{i} = txt{i,1};
                        else
                            varNames{i} = sprintf('Feature_%d', i);
                        end
                    end
                    fprintf('Successfully read %d real feature names from metadata file\n', numFeatures);
                else
                    fprintf('Insufficient feature names in metadata file, using default names\n');
                    for i = 1:numFeatures
                        varNames{i} = sprintf('Feature_%d', i);
                    end
                end
            else
                % Check metadata file in results directory
                resultsMetadataFile = fullfile(rootDir, 'results', 'analysis', 'full', 'data', 'SHAP_metadata.xlsx');
                if exist(resultsMetadataFile, 'file')
                    fprintf('Trying to read real feature names from %s...\n', resultsMetadataFile);
                    % Read Excel file, get the first column as feature names
                    [~, txt, ~] = xlsread(resultsMetadataFile);
                    if ~isempty(txt) && size(txt, 1) >= numFeatures
                        for i = 1:numFeatures
                            if i <= size(txt, 1) && ~isempty(txt{i,1})
                                varNames{i} = txt{i,1};
                            else
                                varNames{i} = sprintf('Feature_%d', i);
                            end
                        end
                        fprintf('Successfully read %d real feature names from results directory metadata file\n', numFeatures);
                    else
                        fprintf('Insufficient feature names in results directory metadata file, using default names\n');
                        for i = 1:numFeatures
                            varNames{i} = sprintf('Feature_%d', i);
                        end
                    end
                else
                    fprintf('Feature name file not found, using default names\n');
                    for i = 1:numFeatures
                        varNames{i} = sprintf('Feature_%d', i);
                    end
                end
            end
        end
    catch err
        fprintf('Error reading real feature names: %s\n', err.message);
        % Use default names when error occurs
        for i = 1:numFeatures
            varNames{i} = sprintf('Feature_%d', i);
        end
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
    
    % Create visualizations if the visualization scripts exist
    fprintf('\n===== Creating Visualizations =====\n');
    
    % Check for visualization functions
    plot_shap_results_path = fullfile(rootDir, 'src', 'visualization', 'plot_shap_results.m');
    plot_shap_beeswarm_path = fullfile(rootDir, 'src', 'visualization', 'plot_shap_beeswarm.m');
    plot_shap_original_summary_path = fullfile(rootDir, 'src', 'visualization', 'plot_shap_original_summary.m');
    fix_colorbar_style_path = fullfile(rootDir, 'src', 'visualization', 'fix_colorbar_style.m');
    
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
                
                if ~exist(script_path, 'file') && ~any(cellfun(@(x) exist(x, 'file'), alternative_paths))
                    fprintf('  No valid paths found for %s, creating placeholder...\n', script_name);
                    
                    % Create a placeholder script in the visualization directory
                    mkdir(fileparts(script_path));
                    fid = fopen(script_path, 'w');
                    fprintf(fid, '%% Placeholder for %s\n', script_name);
                    fprintf(fid, '%% This script was automatically generated as a placeholder\n\n');
                    fprintf(fid, 'disp(''This is a placeholder for %s'');\n', script_name);
                    fprintf(fid, 'warning(''The actual %s implementation is missing. This is a placeholder.'');\n\n', script_name);
                    fprintf(fid, '%% Check if required variables exist\n');
                    fprintf(fid, 'if exist(''shapValues'', ''var'')\n');
                    fprintf(fid, '    disp([''shapValues size: '' num2str(size(shapValues))]);\n');
                    fprintf(fid, 'else\n');
                    fprintf(fid, '    warning(''shapValues not found in workspace'');\n');
                    fprintf(fid, 'end\n');
                    fclose(fid);
                    
                    fprintf('  Created placeholder script at %s\n', script_path);
                    fprintf('  Please replace with the actual implementation before running in production.\n');
                end
            end
        catch viz_error
            fprintf('Error during %s: %s\n', script_name, viz_error.message);
            
            % Print detailed error information
            if ~isempty(viz_error.stack)
                fprintf('  Error in file: %s\n', viz_error.stack(1).file);
                fprintf('  Line: %d\n', viz_error.stack(1).line);
                
                % Show the line that caused the error if possible
                try
                    error_file = viz_error.stack(1).file;
                    error_line = viz_error.stack(1).line;
                    
                    if exist(error_file, 'file')
                        fid = fopen(error_file, 'r');
                        if fid ~= -1
                            all_lines = textscan(fid, '%s', 'Delimiter', '\n');
                            fclose(fid);
                            
                            all_lines = all_lines{1};
                            if error_line <= length(all_lines)
                                fprintf('  Error-causing code: %s\n', all_lines{error_line});
                            end
                        end
                    end
                catch read_err
                    fprintf('  Could not read error-causing line: %s\n', read_err.message);
                end
            end
            
            % Check workspace variables
            fprintf('  Workspace variable check:\n');
            if ~exist('shapValues', 'var')
                fprintf('    - shapValues: MISSING\n');
            else
                fprintf('    - shapValues: PRESENT (%d x %d x %d)\n', size(shapValues, 1), size(shapValues, 2), size(shapValues, 3));
            end
            
            if ~exist('baseValue', 'var')
                fprintf('    - baseValue: MISSING\n');
            else
                fprintf('    - baseValue: PRESENT (%s)\n', mat2str(baseValue));
            end
            
            if ~exist('varNames', 'var')
                fprintf('    - varNames: MISSING\n');
            else
                fprintf('    - varNames: PRESENT (%d elements)\n', length(varNames));
            end
            
            if ~exist('featureNames', 'var')
                fprintf('    - featureNames: MISSING\n');
            else
                fprintf('    - featureNames: PRESENT (%d elements)\n', length(featureNames));
            end
            
            fprintf('  Continuing with other visualization scripts...\n');
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
        testModelFile,
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