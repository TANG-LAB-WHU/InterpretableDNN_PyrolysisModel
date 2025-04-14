%% Export SHAP Results to Excel
% This script exports SHAP analysis results to Excel files for further analysis
% It creates multiple Excel files for different components of the SHAP analysis:
% 1. SHAP values for each sample and feature
% 2. Feature importance (mean absolute SHAP)
% 3. Base values and metadata
%
% All files are saved in the Data directory of the SHAP results folder

fprintf('\n===== Starting SHAP Excel Export =====\n');

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
fprintf('Processing SHAP values for Excel export...\n');

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
if ~exist('shapValues', 'var') || ~exist('baseValue', 'var')
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

%% 1. Create SHAP_metadata.xlsx
fprintf('Creating SHAP_metadata.xlsx...\n');
metadataFile = fullfile(dataDir, 'SHAP_metadata.xlsx');

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

% Write to Excel
writetable(metadataTable, metadataFile, 'Sheet', 'FeatureMetadata');
fprintf('SHAP metadata exported to: %s\n', metadataFile);

%% 2. Export SHAP values for each target to separate Excel files
fprintf('Exporting SHAP values for each target...\n');

% Create a summary file for all targets
summaryFile = fullfile(dataDir, 'SHAP_summary_all_targets.xlsx');

% For each target, create a separate Excel file with SHAP values
for targetIdx = 1:numTargets
    % Create filename with target name
    targetLabel = targetNames{targetIdx};
    outputFile = fullfile(dataDir, ['SHAP_values_' targetLabel '.xlsx']);
    fprintf('Creating Excel file for target: %s\n', targetLabel);
    
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
    
    % Write importance table to Excel
    fprintf('Writing feature importance sheet for target: %s\n', targetLabel);
    writetable(importanceTable, outputFile, 'Sheet', 'FeatureImportance');
    
    % Also add to summary file
    writetable(importanceTable, summaryFile, 'Sheet', ['Importance_' targetLabel]);
    
    % Create a table with SHAP values for each sample
    fprintf('Writing SHAP values sheet for target: %s\n', targetLabel);
    shapTable = array2table(targetSHAP, 'VariableNames', featureNames);
    
    % Add sample ID column if available
    if hasInputData && size(input, 2) == size(targetSHAP, 1)
        sampleIds = (1:size(targetSHAP, 1))';
        shapTable = addvars(shapTable, sampleIds, 'Before', 1, 'NewVariableNames', {'SampleID'});
    end
    
    % Write SHAP values table to Excel
    writetable(shapTable, outputFile, 'Sheet', 'SHAPValues');
    
    % Create a summary sheet with base value and metadata
    fprintf('Writing summary sheet for target: %s\n', targetLabel);
    infoTable = table();
    infoTable.Property = {'TargetName'; 'BaseValue'; 'NumSamples'; 'NumFeatures'};
    infoTable.Value = {targetLabel; baseValue(targetIdx); numInstances; numFeatures};
    
    % Write summary table to Excel
    writetable(infoTable, outputFile, 'Sheet', 'Summary');
    
    fprintf('SHAP values for target %s exported to: %s\n', targetLabel, outputFile);
end

%% 3. Create a combined summary file
fprintf('Creating combined SHAP summary file...\n');

% Add overall information sheet
infoTable = table();
infoTable.Property = {'NumSamples'; 'NumFeatures'; 'NumTargets'};
infoTable.Value = {numInstances; numFeatures; numTargets};

writetable(infoTable, summaryFile, 'Sheet', 'SummaryInfo');

% Add target names sheet
targetTable = table();
targetTable.TargetIndex = (1:numTargets)';
targetTable.TargetName = string(targetNames');
targetTable.BaseValue = baseValue';

writetable(targetTable, summaryFile, 'Sheet', 'Targets');

fprintf('Combined SHAP summary exported to: %s\n', summaryFile);

% Also create a file with the standard name shap_analysis_results.xlsx
analysisResultsFile = fullfile(shapDir, 'shap_analysis_results.xlsx');
copyfile(summaryFile, analysisResultsFile);
fprintf('Copied summary to standard name: %s\n', analysisResultsFile);

fprintf('===== SHAP Excel Export Completed Successfully =====\n\n'); 