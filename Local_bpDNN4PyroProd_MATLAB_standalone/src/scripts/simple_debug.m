% Simplified Debug Script for Testing Purposes
fprintf('=== Starting Simplified Debug Script ===\n\n');

% Get the root directory of the project
rootDir = fileparts(fileparts(fileparts(mfilename('fullpath'))));
fprintf('Root directory: %s\n', rootDir);

% Define directory paths relative to root directory
scriptDir = fullfile(rootDir, 'src', 'scripts');
modelDir = fullfile(rootDir, 'src', 'model'); 
resultsDir = fullfile(rootDir, 'results');
debugDir = fullfile(rootDir, 'debug_dir');
bestModelDir = fullfile(resultsDir, 'best_model');
optimizationDir = fullfile(resultsDir, 'optimization');
analysisDir = fullfile(resultsDir, 'analysis', 'debug');
analysisDataDir = fullfile(analysisDir, 'data');
analysisFigDir = fullfile(analysisDir, 'figures');

% Print all the directory paths
fprintf('\n--- Directory Paths ---\n');
fprintf('Script directory: %s\n', scriptDir);
fprintf('Model directory: %s\n', modelDir);
fprintf('Results directory: %s\n', resultsDir);
fprintf('Debug directory: %s\n', debugDir);
fprintf('Best model directory: %s\n', bestModelDir);
fprintf('Optimization directory: %s\n', optimizationDir);
fprintf('Analysis directory: %s\n', analysisDir);
fprintf('Analysis data directory: %s\n', analysisDataDir);
fprintf('Analysis figures directory: %s\n', analysisFigDir);

% Check if directories exist and create them if they don't
fprintf('\n--- Creating Missing Directories ---\n');
dirs_to_check = {debugDir, analysisDir, analysisDataDir, analysisFigDir};
for i = 1:length(dirs_to_check)
    if ~exist(dirs_to_check{i}, 'dir')
        fprintf('Creating directory: %s\n', dirs_to_check{i});
        mkdir(dirs_to_check{i});
    else
        fprintf('Directory already exists: %s\n', dirs_to_check{i});
    end
end

% Create a simple data file for testing
fprintf('\n--- Creating Test Data File ---\n');
test_data_file = fullfile(debugDir, 'test_data.mat');
if ~exist(test_data_file, 'file')
    % Create some simple test data
    fprintf('Creating test data...\n');
    x = rand(5, 10);  % 5 features, 10 samples
    y = rand(2, 10);  % 2 outputs, 10 samples
    
    % Save the data
    save(test_data_file, 'x', 'y');
    fprintf('Test data saved to: %s\n', test_data_file);
else
    fprintf('Test data file already exists: %s\n', test_data_file);
    % Load the data to verify
    load(test_data_file, 'x', 'y');
    fprintf('Loaded test data: %d features, %d samples, %d outputs\n', ...
        size(x, 1), size(x, 2), size(y, 1));
end

% Add model directory to path
if ~contains(path, modelDir)
    addpath(modelDir);
    fprintf('Added model directory to path: %s\n', modelDir);
end

% Create a simple log file
fprintf('\n--- Creating Log File ---\n');
log_file = fullfile(debugDir, 'simple_debug_log.txt');
fid = fopen(log_file, 'w');
if fid < 0
    fprintf('ERROR: Could not create log file: %s\n', log_file);
else
    fprintf(fid, 'Simple Debug Script Run\n');
    fprintf(fid, 'Date: %s\n', datestr(now));
    fprintf(fid, 'Root directory: %s\n', rootDir);
    fprintf(fid, 'Test data file: %s\n', test_data_file);
    fprintf(fid, 'Test data size: [%d, %d] features, [%d, %d] outputs\n', ...
        size(x, 1), size(x, 2), size(y, 1), size(y, 2));
    fclose(fid);
    fprintf('Log file created: %s\n', log_file);
end

% Generate a simple script file automatically
fprintf('\n--- Generating Auto Script ---\n');
auto_script_file = fullfile(scriptDir, 'auto_generated.m');
fid = fopen(auto_script_file, 'w');
if fid < 0
    fprintf('ERROR: Could not create auto-generated script: %s\n', auto_script_file);
else
    fprintf(fid, '%% Auto-generated script from simple_debug.m\n');
    fprintf(fid, 'fprintf(''Auto-generated script executed!\\n'');\n');
    fprintf(fid, 'disp(''Generated on: %s'');\n', datestr(now));
    fprintf(fid, 'disp(''This is a test file!'');\n');
    fclose(fid);
    fprintf('Auto-generated script created: %s\n', auto_script_file);
end

fprintf('\n=== Simple Debug Script Completed ===\n'); 