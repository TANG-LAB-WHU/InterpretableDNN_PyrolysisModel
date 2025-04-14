%% SHAP Analysis Master Script
% This script runs the complete SHAP analysis workflow
% 1. Calculate SHAP values
% 2. Generate all visualizations
% 3. Export results to Excel
%
% All results will be saved in the results/analysis directory

% Only clear if not called from another script
if ~exist('parentScript', 'var')
    % Close figures but don't clear workspace when called from debug_run_analysis
    % clear;
    close all;
    
    % Define paths
    scriptPath = mfilename('fullpath');
    [scriptDir, ~, ~] = fileparts(scriptPath);
    rootDir = fileparts(scriptDir);
    resultsDir = fullfile(rootDir, 'results');
    
    % Ask user if they want to run in debug mode (smaller dataset) or full analysis
    prompt = 'Run in debug mode (y/n - debug uses fewer samples for faster execution): ';
    userInput = input(prompt, 's');
    if strcmpi(userInput, 'y') || strcmpi(userInput, 'yes')
        analysisMode = 'debug';
        fprintf('Running in debug mode with reduced samples.\n');
    else
        analysisMode = 'full';
        fprintf('Running full analysis on complete dataset.\n');
    end
    
    % Create analysis directory
    analysisDir = fullfile(resultsDir, 'analysis', analysisMode);
    if ~exist(analysisDir, 'dir')
        mkdir(analysisDir);
    end
    
    % Create subdirectories for figures and data
    figDir = fullfile(analysisDir, 'figures');
    if ~exist(figDir, 'dir')
        mkdir(figDir);
    end
    
    dataDir = fullfile(analysisDir, 'data');
    if ~exist(dataDir, 'dir')
        mkdir(dataDir);
    end
    
    % Define the output paths only if not already defined
    if ~exist('shapDir', 'var')
        shapDir = analysisDir;
    end
else
    % When called from another script, just make sure required directories exist
    if ~exist('shapDir', 'var')
        error('shapDir must be defined when calling run_shap_analysis from another script');
    end
    
    % Ensure figures and data directories exist
    figDir = fullfile(shapDir, 'figures');
    if ~exist(figDir, 'dir')
        mkdir(figDir);
    end
    
    dataDir = fullfile(shapDir, 'data');
    if ~exist(dataDir, 'dir')
        mkdir(dataDir);
    end
end

fprintf('Results will be saved in: %s\n', shapDir);

% Start timer
totalStartTime = tic;
fprintf('=== Starting SHAP Analysis (%s mode) ===\n', analysisMode);

% Step 1: Calculate SHAP values
fprintf('\n=== Step 1: Calculating SHAP Values ===\n');
stepStartTime = tic;

% Check if SHAP results already exist
shap_results_file = fullfile(dataDir, 'shap_results.mat');
shap_results_exist = exist(shap_results_file, 'file');

if shap_results_exist
    fprintf('SHAP results file already exists at: %s\n', shap_results_file);
    fprintf('Checking if file contains required variables...\n');
    
    % Check if file contains required variables
    info = whos('-file', shap_results_file);
    var_names = {info.name};
    
    % Check for required variables
    has_shap_values = ismember('shapValues', var_names);
    has_base_value = ismember('baseValue', var_names);
    
    if has_shap_values && has_base_value
        fprintf('Existing SHAP results file contains required variables. Loading...\n');
        % Load variables selectively to avoid overwriting existing workspace variables
        if ~exist('shapValues', 'var') || ~exist('baseValue', 'var')
            results = load(shap_results_file, 'shapValues', 'baseValue');
            if ~exist('shapValues', 'var') && isfield(results, 'shapValues')
                shapValues = results.shapValues;
            end
            if ~exist('baseValue', 'var') && isfield(results, 'baseValue')
                baseValue = results.baseValue;
            end
            % Load other potentially useful variables if they don't exist in workspace
            if ~exist('varNames', 'var') && ismember('varNames', var_names)
                results = load(shap_results_file, 'varNames');
                varNames = results.varNames;
            end
            if ~exist('featureNames', 'var') && ismember('featureNames', var_names)
                results = load(shap_results_file, 'featureNames');
                featureNames = results.featureNames;
            end
            if ~exist('targetNames', 'var') && ismember('targetNames', var_names)
                results = load(shap_results_file, 'targetNames');
                targetNames = results.targetNames;
            end
        end
        fprintf('✓ Successfully loaded existing SHAP results.\n');
    else
        fprintf('Existing SHAP results file is incomplete. Attempting to calculate new values...\n');
        shap_results_exist = false;
    end
end

% Run the script to calculate SHAP values if needed
if ~shap_results_exist || ~has_shap_values || ~has_base_value
    try
        fprintf('Loading model and calculating SHAP values...\n');
        calc_shap_values;
        fprintf('✓ SHAP values calculated successfully!\n');
    catch ME
        fprintf('✗ Error calculating SHAP values: %s\n', ME.message);
        
        % Check if we already loaded results earlier
        if exist('shapValues', 'var') && exist('baseValue', 'var')
            fprintf('Using previously loaded SHAP values to continue analysis.\n');
        else
            % Try one more time to load from file
            if exist(shap_results_file, 'file')
                try
                    fprintf('Attempting to load existing SHAP results despite calculation error...\n');
                    results = load(shap_results_file);
                    if isfield(results, 'shapValues') && isfield(results, 'baseValue')
                        fprintf('Successfully loaded essential SHAP variables from file.\n');
                        shapValues = results.shapValues;
                        baseValue = results.baseValue;
                        if isfield(results, 'varNames')
                            varNames = results.varNames;
                        end
                        if isfield(results, 'featureNames')
                            featureNames = results.featureNames;
                        end
                    else
                        fprintf('Failed to load required variables. Analysis cannot continue.\n');
                        fprintf('Error details:\n%s\n', getReport(ME));
                        return;
                    end
                catch load_err
                    fprintf('Failed to load SHAP results: %s\n', load_err.message);
                    fprintf('Error details:\n%s\n', getReport(ME));
                    fprintf('Unable to proceed with visualization and export steps.\n');
                    return;
                end
            else
                fprintf('No existing SHAP results found. Unable to proceed with analysis.\n');
                fprintf('Error details:\n%s\n', getReport(ME));
                return;
            end
        end
    end
end

fprintf('Step 1 completed in %.2f seconds.\n', toc(stepStartTime));

% Step 2: Generate visualizations
fprintf('\n=== Step 2: Generating Visualizations ===\n');
stepStartTime = tic;

% Each visualization is tried independently to ensure all possible visualizations are created
% Run the beeswarm plot
success_count = 0;
try
    % Original beeswarm plot
    fprintf('Creating beeswarm plots...\n');
    plot_shap_beeswarm;
    fprintf('✓ Beeswarm plots created successfully!\n');
    success_count = success_count + 1;
catch ME
    fprintf('✗ Error generating beeswarm plots: %s\n', ME.message);
end

% Run the original summary plots
try
    % Original summary plots
    fprintf('Creating original summary plots...\n');
    plot_shap_original_summary;
    fprintf('✓ Original summary plots created successfully!\n');
    success_count = success_count + 1;
catch ME
    fprintf('✗ Error generating original summary plots: %s\n', ME.message);
end

% Run the enhanced result plots
try
    % Enhanced result plots
    fprintf('Creating enhanced visualization plots...\n');
    plot_shap_results;
    fprintf('✓ Enhanced visualization plots created successfully!\n');
    success_count = success_count + 1;
catch ME
    fprintf('✗ Error generating enhanced visualization plots: %s\n', ME.message);
    fprintf('Error details:\n%s\n', getReport(ME));
end

if success_count > 0
    fprintf('✓ %d of 3 visualization types created successfully.\n', success_count);
else
    fprintf('✗ All visualizations failed. Continuing with export step...\n');
end

fprintf('Step 2 completed in %.2f seconds.\n', toc(stepStartTime));

% Step 3: Export to Excel
fprintf('\n=== Step 3: Exporting Results to Excel ===\n');
stepStartTime = tic;

try
    fprintf('Exporting SHAP data to Excel files...\n');
    export_shap_to_excel;
    fprintf('✓ SHAP data exported successfully!\n');
catch ME
    fprintf('✗ Error exporting to Excel: %s\n', ME.message);
    fprintf('Error details:\n%s\n', getReport(ME));
end

fprintf('Step 3 completed in %.2f seconds.\n', toc(stepStartTime));

% Calculate total execution time
totalTime = toc(totalStartTime);
fprintf('\n=== SHAP Analysis Completed ===\n');
fprintf('Total execution time: %.2f seconds (%.2f minutes)\n', totalTime, totalTime/60);

% Print summary of output locations
fprintf('\nResults can be found in the following locations:\n');
fprintf('- Figures: %s\n', figDir);
fprintf('- Data files: %s\n', dataDir);
fprintf('- Excel exports: %s\n', fullfile(dataDir));

% Check for expected outputs and report
figFiles = dir(fullfile(figDir, '*.png'));
dataFiles = dir(fullfile(dataDir, '*.xlsx'));
fprintf('\nOutput file summary:\n');
fprintf('- Figure files (.png): %d\n', length(figFiles));
fprintf('- Excel files (.xlsx): %d\n', length(dataFiles));

if length(dataFiles) == 0
    fprintf('\nWARNING: No Excel files were generated! There may be issues with the export_shap_to_excel.m script.\n');
end

if length(figFiles) == 0
    fprintf('\nWARNING: No figure files were generated! There may be issues with the visualization scripts.\n');
end

% Display completion message
fprintf('\nSHAP analysis complete! You can now review the results in the output directories.\n'); 