%% Export SHAP Results to Excel
% This script exports SHAP analysis results to Excel files for further analysis
% It creates multiple Excel files for different components of the SHAP analysis:
% 1. SHAP values for each sample and feature
% 2. Feature importance (mean absolute SHAP)
% 3. Base values and metadata
%
% All files are saved in the Data directory of the SHAP results folder
% Enhanced with ICC cluster compatibility - CSV export for environments without Excel support

fprintf('\n===== Starting SHAP Export =====\n');

% Check if shapDir exists in the workspace
if ~exist('shapDir', 'var')
    % Get the script's directory
    scriptPath = mfilename('fullpath');
    [scriptDir, ~, ~] = fileparts(scriptPath);
    
    % Set default SHAP directory
    rootDir = fileparts(fileparts(scriptDir)); 
    resultsDir = fullfile(rootDir, 'results');
    shapDir = fullfile(resultsDir, 'analysis', 'debug');
    
    % Try to determine which analysis mode we're in based on environment
    dbstack_info = dbstack;
    if length(dbstack_info) > 1
        caller_name = dbstack_info(2).name;
        if contains(lower(caller_name), 'debug')
            shapDir = fullfile(rootDir, 'results', 'analysis', 'debug');
            fprintf('Detected debug mode. Using shapDir: %s\n', shapDir);
        else
            shapDir = fullfile(rootDir, 'results', 'analysis', 'full');
            fprintf('Detected full analysis mode. Using shapDir: %s\n', shapDir);
        end
    end
    
    warning('shapDir not found in workspace, using default path: %s', shapDir);
end
fprintf('Using SHAP directory: %s\n', shapDir);

% Create data subfolder in shapDir if it doesn't exist
dataDir = fullfile(shapDir, 'data');
if ~exist(dataDir, 'dir')
    mkdir(dataDir);
    fprintf('Created data directory: %s\n', dataDir);
else
    fprintf('Using existing data directory: %s\n', dataDir);
end

% Load SHAP results for export
fprintf('Processing SHAP values for export...\n');

% Load SHAP data
shap_file = fullfile(dataDir, 'shap_results.mat');
fprintf('Checking for SHAP results file: %s\n', shap_file);
if ~exist(shap_file, 'file')
    fprintf('SHAP results file not found. File: %s\n', shap_file);
    fprintf('Looking for alternative files in data directory...\n');
    
    % Check for any .mat files in data directory
    matFiles = dir(fullfile(dataDir, '*.mat'));
    if ~isempty(matFiles)
        fprintf('Found %d .mat files in data directory:\n', length(matFiles));
        for i = 1:length(matFiles)
            fprintf('  %s\n', matFiles(i).name);
        end
        
        % Try using the first one
        shap_file = fullfile(dataDir, matFiles(1).name);
        fprintf('Attempting to use first mat file: %s\n', shap_file);
    else
        error('No SHAP results files found in %s. Run calc_shap_values.m first.', dataDir);
    end
end

% Check if required variables already exist in workspace
if ~exist('shapValues', 'var') || !exist('baseValue', 'var')
    fprintf('Required SHAP variables not found in workspace. Loading from %s\n', shap_file);
    % Load with specific variable names to ensure they're all available
    results = load(shap_file);
    if ~isfield(results, 'shapValues') || ~isfield(results, 'baseValue')
        error('Missing required variables in SHAP results. Re-run calc_shap_values.m.');
    end
    
    % Extract variables from the loaded structure
    shapValues = results.shapValues;
    baseValue = results.baseValue;
    
    % Check if we have input data
    hasInputData = isfield(results, 'input');
    if hasInputData
        input = results.input;
    else
        warning('Input data not found in SHAP results. Some exports will be limited.');
        hasInputData = false;
    end
    
    % Setup feature names
    if isfield(results, 'featureNames')
        % First try to use featureNames if available (preferred)
        featureNames = results.featureNames;
        fprintf('Using featureNames from SHAP results file (%d names)\n', length(featureNames));
    elseif isfield(results, 'varNames')
        % Fall back to varNames if featureNames not available
        featureNames = results.varNames;
        fprintf('Using varNames from SHAP results file as featureNames (%d names)\n', length(featureNames));
    else
        % Create default feature names if not provided
        featureNames = cell(1, size(shapValues, 2));
        for i = 1:size(shapValues, 2)
            featureNames{i} = sprintf('Feature %d', i);
        end
        disp('Using default feature names (no featureNames or varNames found in SHAP results)');
    end
    
    % Setup target names
    if isfield(results, 'targetNames')
        targetNames = results.targetNames;
        fprintf('Using targetNames from SHAP results file (%d names)\n', length(targetNames));
    else
        % Create default target names if not provided
        if ndims(shapValues) == 3
            targetNames = cell(1, size(shapValues, 3));
            % First try the standard pyrolysis product names
            if size(shapValues, 3) == 3
                targetNames{1} = 'Char/%';
                targetNames{2} = 'Liquid/%';
                targetNames{3} = 'Gas/%';
                fprintf('Using standard pyrolysis product names for targets\n');
            else
                for i = 1:size(shapValues, 3)
                    targetNames{i} = sprintf('Target %d', i);
                end
                fprintf('Using default target names (no targetNames found)\n');
            end
        else
            targetNames = {'Target 1'};
            fprintf('Using default target name for single output\n');
        end
    end
else
    fprintf('Using SHAP variables from current workspace\n');
    
    % Make sure we have feature names
    if ~exist('featureNames', 'var')
        if exist('varNames', 'var')
            featureNames = varNames;
            fprintf('Using varNames from workspace as featureNames\n');
        else
            % Create default feature names if not provided
            featureNames = cell(1, size(shapValues, 2));
            for i = 1:size(shapValues, 2)
                featureNames{i} = sprintf('Feature %d', i);
            end
            disp('Using default feature names (no featureNames or varNames found in workspace)');
        end
    else
        fprintf('Using featureNames from workspace\n');
    end
    
    % Make sure we have target names
    if ~exist('targetNames', 'var')
        % Create default target names if not provided
        if ndims(shapValues) == 3
            targetNames = cell(1, size(shapValues, 3));
            % First try the standard pyrolysis product names
            if size(shapValues, 3) == 3
                targetNames{1} = 'Char/%';
                targetNames{2} = 'Liquid/%';
                targetNames{3} = 'Gas/%';
                fprintf('Using standard pyrolysis product names for targets\n');
            else
                for i = 1:size(shapValues, 3)
                    targetNames{i} = sprintf('Target %d', i);
                end
                fprintf('Using default target names (no targetNames found)\n');
            end
        else
            targetNames = {'Target 1'};
            fprintf('Using default target name for single output\n');
        end
    else
        fprintf('Using targetNames from workspace\n');
    end
    
    % Check if we have input data
    hasInputData = exist('input', 'var');
    if ~hasInputData
        warning('Input data not found in workspace. Some exports will be limited.');
    end
end

% Determine dimensions of SHAP values
if ndims(shapValues) == 3
    % We have SHAP values for multiple targets
    [numInstances, numFeatures, numTargets] = size(shapValues);
    fprintf('Found 3D SHAP values with %d instances, %d features, and %d targets\n', ...
        numInstances, numFeatures, numTargets);
else
    % We have SHAP values for a single target 
    [numInstances, numFeatures] = size(shapValues);
    numTargets = 1;
    % Reshape to make it 3D for consistent handling
    shapValues = reshape(shapValues, [numInstances, numFeatures, 1]);
    fprintf('Found 2D SHAP values with %d instances and %d features\n', ...
        numInstances, numFeatures);
end

% Ensure feature names matches the feature count
if length(featureNames) < numFeatures
    % Add default names for any missing
    for i = length(featureNames)+1:numFeatures
        featureNames{i} = sprintf('Feature %d', i);
    end
    disp('Extended feature names to match feature count');
elseif length(featureNames) > numFeatures
    % Truncate to match actual feature count
    featureNames = featureNames(1:numFeatures);
    disp('Truncated feature names to match feature count');
end

% Check for duplicate feature names and make them unique
[~, uniqueIndices] = unique(featureNames);
if length(uniqueIndices) < length(featureNames)
    dupIndices = setdiff(1:length(featureNames), uniqueIndices);
    disp(['Found ' num2str(length(dupIndices)) ' duplicate feature names. Making them unique...']);
    
    % Create a map to track occurrences of each name
    nameCountMap = containers.Map('KeyType', 'char', 'ValueType', 'double');
    
    for i = 1:length(featureNames)
        currentName = featureNames{i};
        if ~nameCountMap.isKey(currentName)
            nameCountMap(currentName) = 1;
        else
            % If the name already exists, append a unique suffix
            count = nameCountMap(currentName);
            featureNames{i} = [currentName '_' num2str(count)];
            nameCountMap(currentName) = count + 1;
        end
    end
    
    disp('Feature names have been made unique by appending indices to duplicates.');
end

% Ensure target names match target count
if length(targetNames) < numTargets
    % Add default names for any missing
    for i = length(targetNames)+1:numTargets
        targetNames{i} = sprintf('Target %d', i);
    end
    disp('Extended target names to match target count');
elseif length(targetNames) > numTargets
    % Truncate to match actual target count
    targetNames = targetNames(1:numTargets);
    disp('Truncated target names to match target count');
end

%% Detect if we're running on a headless environment like the ICC cluster
% Check if we're on a Linux system and try to determine if Excel operations might fail
isICC = false;
isHeadless = false;

try
    % Check if we're on Linux
    if isunix
        % Additional check for ICC cluster (looking at hostname or path)
        [~, hostname] = system('hostname');
        if contains(lower(hostname), 'icc') || contains(lower(hostname), 'iccompute') || ...
           contains(pwd, '/projects/illinois') || contains(computer('arch'), 'glnxa64')
            isICC = true;
            fprintf('Detected ICC cluster environment.\n');
        end
        
        % Check for display - might indicate headless environment
        [status, ~] = system('echo $DISPLAY');
        if status ~= 0 || isempty(getenv('DISPLAY'))
            isHeadless = true;
            fprintf('Detected headless environment. Excel operations may fail.\n');
        end
    end
catch
    % If any error occurs during detection, assume we might be on a restricted environment
    fprintf('Error during environment detection. Using cautious export approach.\n');
    isHeadless = true;
end

% Select export strategy based on environment
useCSVFallback = isICC || isHeadless;

if useCSVFallback
    fprintf('Using CSV export as fallback for compatibility with headless environment.\n');
end

%% Export SHAP metadata
fprintf('Creating SHAP metadata file...\n');
metadataFile = fullfile(dataDir, 'SHAP_metadata');

% Create an empty table with the correct number of rows
metadataTable = table();
metadataTable.FeatureNames = string(featureNames');
metadataTable.DescriptionCol = cell(numFeatures, 1);
metadataTable.DataType = cell(numFeatures, 1);
metadataTable.Units = cell(numFeatures, 1);

% Try to extract units from feature names
for i = 1:numFeatures
    featureName = featureNames{i};
    % Look for /%
    if contains(featureName, '/%')
        metadataTable.Units{i} = '%';
    elseif contains(featureName, '/Celsius')
        metadataTable.Units{i} = 'Celsius';
    elseif contains(featureName, '/min')
        metadataTable.Units{i} = 'min';
    elseif contains(featureName, '/(K/min)')
        metadataTable.Units{i} = 'K/min';
    else
        metadataTable.Units{i} = '';
    end
    
    % Generate descriptions based on feature names
    metadataTable.DescriptionCol{i} = ['Feature: ' featureNames{i}];
    
    % Try to guess data type
    if contains(featureName, 'ID') || contains(featureName, 'Type') || contains(featureName, 'Location')
        metadataTable.DataType{i} = 'categorical';
    else
        metadataTable.DataType{i} = 'numeric';
    end
end

% Ensure all columns have the correct data type
for i = 1:size(metadataTable, 1)
    % Ensure DescriptionCol is initialized as cell
    if isempty(metadataTable.DescriptionCol{i})
        metadataTable.DescriptionCol{i} = '';
    end
    if isempty(metadataTable.DataType{i})
        metadataTable.DataType{i} = '';
    end
    if isempty(metadataTable.Units{i})
        metadataTable.Units{i} = '';
    end
end

% Write to file
try
    if ~useCSVFallback
        % Use Excel if not in headless/ICC environment
        writetable(metadataTable, [metadataFile '.xlsx'], 'Sheet', 'FeatureMetadata');
        fprintf('SHAP metadata exported to: %s.xlsx\n', metadataFile);
    catch excelErr
        % If Excel export fails, fall back to CSV
        fprintf('Excel export failed: %s\n', excelErr.message);
        useCSVFallback = true;
        fprintf('Falling back to CSV export...\n');
    end
catch
    % Generic error handler
    useCSVFallback = true;
    fprintf('Error during Excel export, falling back to CSV...\n');
end

% Always export CSV version for compatibility
if useCSVFallback
    writetable(metadataTable, [metadataFile '.csv']);
    fprintf('SHAP metadata exported to: %s.csv\n', metadataFile);
end

%% Create a summary file for all targets
fprintf('Creating SHAP summary for all targets...\n');
summaryFile = fullfile(dataDir, 'SHAP_summary_all_targets');
summaryTables = {};
summarySheetNames = {};

% For each target, export SHAP values
for targetIdx = 1:numTargets
    % Create filename with target name
    targetLabel = strrep(targetNames{targetIdx}, '%', 'pct'); % Replace % with pct for filenames
    targetLabel = strrep(targetLabel, '/', '_');  % Replace / with _ for filenames
    outputFile = fullfile(dataDir, ['SHAP_values_' targetLabel]);
    fprintf('Processing data for target: %s\n', targetNames{targetIdx});
    
    % Extract SHAP values for this target
    targetSHAP = shapValues(:, :, targetIdx);
    
    % Calculate mean absolute SHAP values (feature importance)
    featureImportance = mean(abs(targetSHAP), 1)';
    [sortedImportance, sortedIdx] = sort(featureImportance, 'descend');
    sortedFeatureNames = featureNames(sortedIdx);
    
    % Create a table with importance values
    importanceTable = table();
    importanceTable.FeatureName = string(sortedFeatureNames');
    importanceTable.Importance = sortedImportance;
    importanceTable.RelativeImportance = sortedImportance / max(sortedImportance) * 100;
    
    % Add to summary tables collection
    summaryTables{end+1} = importanceTable;
    summarySheetNames{end+1} = ['Importance_' strrep(targetLabel, ' ', '_')];
    
    % Create a table with SHAP values for each sample
    shapTable = array2table(targetSHAP, 'VariableNames', featureNames);
    
    % Add sample ID column if available
    if hasInputData && size(input, 2) == size(targetSHAP, 1)
        sampleIds = (1:size(targetSHAP, 1))';
        shapTable = addvars(shapTable, sampleIds, 'Before', 1, 'NewVariableNames', {'SampleID'});
    end
    
    % Create a summary information table
    infoTable = table();
    infoTable.Property = {'TargetName'; 'BaseValue'; 'NumSamples'; 'NumFeatures'};
    infoTable.Value = {targetNames{targetIdx}; baseValue(targetIdx); numInstances; numFeatures};
    
    % Write target data to files
    try
        if ~useCSVFallback
            % Try Excel export
            writetable(importanceTable, [outputFile '.xlsx'], 'Sheet', 'FeatureImportance');
            writetable(shapTable, [outputFile '.xlsx'], 'Sheet', 'SHAPValues', 'WriteMode', 'append');
            writetable(infoTable, [outputFile '.xlsx'], 'Sheet', 'Summary', 'WriteMode', 'append');
            fprintf('SHAP values for target %s exported to: %s.xlsx\n', targetNames{targetIdx}, outputFile);
        catch excelErr
            fprintf('Excel export failed for target %s: %s\n', targetNames{targetIdx}, excelErr.message);
            useCSVFallback = true;
        end
    catch
        useCSVFallback = true;
        fprintf('Error during Excel export, falling back to CSV...\n');
    end
    
    % Export CSV versions for compatibility or as fallback
    if useCSVFallback
        writetable(importanceTable, [outputFile '_importance.csv']);
        writetable(shapTable, [outputFile '_values.csv']);
        writetable(infoTable, [outputFile '_summary.csv']);
        fprintf('SHAP data for target %s exported as CSV files\n', targetNames{targetIdx});
    end
end

% Create overall information table
infoTable = table();
infoTable.Property = {'NumSamples'; 'NumFeatures'; 'NumTargets'};
infoTable.Value = {numInstances; numFeatures; numTargets};

% Add target names table
targetTable = table();
targetTable.TargetIndex = (1:numTargets)';
targetTable.TargetName = string(targetNames');
targetTable.BaseValue = baseValue';

% Export summary data
try
    if ~useCSVFallback
        % Try exporting summary to Excel
        writetable(infoTable, [summaryFile '.xlsx'], 'Sheet', 'SummaryInfo');
        writetable(targetTable, [summaryFile '.xlsx'], 'Sheet', 'Targets', 'WriteMode', 'append');
        
        % Write each importance table to a separate sheet
        for i = 1:length(summaryTables)
            writetable(summaryTables{i}, [summaryFile '.xlsx'], 'Sheet', summarySheetNames{i}, 'WriteMode', 'append');
        end
        fprintf('Combined SHAP summary exported to: %s.xlsx\n', summaryFile);
        
        % Also create standard name file
        analysisResultsFile = fullfile(shapDir, 'shap_analysis_results.xlsx');
        copyfile([summaryFile '.xlsx'], analysisResultsFile);
        fprintf('Copied summary to standard name: %s\n', analysisResultsFile);
    catch excelErr
        fprintf('Excel export for summary file failed: %s\n', excelErr.message);
        useCSVFallback = true;
    end
catch
    useCSVFallback = true;
    fprintf('Error during Excel export, falling back to CSV...\n');
end

% Export CSV versions of summary data
if useCSVFallback
    writetable(infoTable, [summaryFile '_info.csv']);
    writetable(targetTable, [summaryFile '_targets.csv']);
    
    % Export each importance table to a separate CSV
    for i = 1:length(summaryTables)
        writetable(summaryTables{i}, [summaryFile '_' summarySheetNames{i} '.csv']);
    end
    fprintf('Combined SHAP summary exported as CSV files\n');
    
    % Create a simple text file as the standard result file
    standardResultsFile = fullfile(shapDir, 'shap_analysis_results.txt');
    fid = fopen(standardResultsFile, 'w');
    if fid > 0
        fprintf(fid, 'SHAP Analysis Results\n');
        fprintf(fid, '===================\n\n');
        fprintf(fid, 'Number of Samples: %d\n', numInstances);
        fprintf(fid, 'Number of Features: %d\n', numFeatures);
        fprintf(fid, 'Number of Targets: %d\n\n', numTargets);
        fprintf(fid, 'Target Names:\n');
        for i = 1:numTargets
            fprintf(fid, '  %d. %s (Base Value: %.6f)\n', i, targetNames{i}, baseValue(i));
        end
        fprintf(fid, '\nResults exported as CSV files in: %s\n', dataDir);
        fclose(fid);
    end
    fprintf('Created standard results text file: %s\n', standardResultsFile);
end

fprintf('===== SHAP Export Completed Successfully =====\n\n');