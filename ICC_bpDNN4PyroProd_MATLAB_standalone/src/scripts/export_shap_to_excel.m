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
    
    % Setup variable names
    if isfield(results, 'varNames')
        varNames = results.varNames;
        featureNames = varNames;
    else
        % Create default feature names if not provided
        featureNames = cell(1, size(shapValues, 2));
        for i = 1:size(shapValues, 2)
            featureNames{i} = sprintf('Feature %d', i);
        end
        disp('Using default feature names (no varNames found in SHAP results)');
    end
    
    % Setup target names
    if isfield(results, 'targetNames')
        targetNames = results.targetNames;
    else
        % Create default target names if not provided
        if ndims(shapValues) == 3
            targetNames = cell(1, size(shapValues, 3));
            for i = 1:size(shapValues, 3)
                targetNames{i} = sprintf('Target %d', i);
            end
        else
            targetNames = {'Target 1'};
        end
        disp('Using default target names (no targetNames found in SHAP results)');
    end
else
    fprintf('Using SHAP variables from current workspace\n');
    
    % Make sure we have feature names
    if ~exist('featureNames', 'var')
        if exist('varNames', 'var')
            featureNames = varNames;
        else
            % Create default feature names if not provided
            featureNames = cell(1, size(shapValues, 2));
            for i = 1:size(shapValues, 2)
                featureNames{i} = sprintf('Feature %d', i);
            end
            disp('Using default feature names (no featureNames or varNames found)');
        end
    end
    
    % Check if we have input data
    hasInputData = exist('input', 'var');
    if ~hasInputData
        warning('Input data not found in workspace. Some exports will be limited.');
    end
    
    % Setup target names if not available
    if ~exist('targetNames', 'var')
        % Create default target names if not provided
        if ndims(shapValues) == 3
            targetNames = cell(1, size(shapValues, 3));
            for i = 1:size(shapValues, 3)
                targetNames{i} = sprintf('Target %d', i);
            end
        else
            targetNames = {'Target 1'};
        end
        disp('Using default target names (no targetNames found in workspace)');
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

% Fill the table data
fprintf('Filling metadata table with %d feature rows...\n', numFeatures);
for i = 1:numFeatures
    metadataTable.DescriptionCol{i} = ['Feature: ' featureNames{i}];
    
    % Determine data type
    if hasInputData && size(input, 1) >= i
        value = input(i, 1);
        
        % Determine data type
        if isinteger(value)
            metadataTable.DataType{i} = 'Integer';
        elseif isfloat(value)
            metadataTable.DataType{i} = 'Float';
        elseif islogical(value)
            metadataTable.DataType{i} = 'Boolean';
        else
            metadataTable.DataType{i} = 'Unknown';
        end
    else
        metadataTable.DataType{i} = 'Unknown';
    end
    
    % Default units
    metadataTable.Units{i} = '-';
end

% Check if all columns have the correct number of rows
fprintf('Metadata table created with %d rows and %d columns.\n', ...
    height(metadataTable), width(metadataTable));

if height(metadataTable) ~= numFeatures
    warning('Metadata table has %d rows but should have %d rows (numFeatures). Adjusting...', ...
        height(metadataTable), numFeatures);
    
    % If row count is insufficient, expand the table
    if height(metadataTable) < numFeatures
        extraRows = numFeatures - height(metadataTable);
        extraTable = table();
        extraTable.FeatureNames = string(cell(extraRows, 1));
        extraTable.DescriptionCol = cell(extraRows, 1);
        extraTable.DataType = cell(extraRows, 1);
        extraTable.Units = cell(extraRows, 1);
        
        % Fill additional rows
        for i = 1:extraRows
            idx = height(metadataTable) + i;
            if idx <= length(featureNames)
                extraTable.FeatureNames(i) = string(featureNames{idx});
                extraTable.DescriptionCol{i} = ['Feature: ' featureNames{idx}];
            else
                extraTable.FeatureNames(i) = string(sprintf('Unknown_Feature_%d', idx));
                extraTable.DescriptionCol{i} = ['Unknown Feature ' num2str(idx)];
            end
            extraTable.DataType{i} = 'Unknown';
            extraTable.Units{i} = '-';
        end
        
        % Combine tables
        metadataTable = [metadataTable; extraTable];
    end
    
    % If too many rows, truncate the table
    if height(metadataTable) > numFeatures
        metadataTable = metadataTable(1:numFeatures, :);
    end
end

% Add model metadata at the end
summaryRows = {
    'ModelType', 'Neural Network Model', 'String', '-';
    'NumSamples', num2str(numInstances), 'Integer', 'count';
    'NumFeatures', num2str(numFeatures), 'Integer', 'count';
    'NumTargets', num2str(numTargets), 'Integer', 'count';
    'AnalysisDate', datestr(now), 'Date', '-'
};
additionalRows = cell2table(summaryRows, 'VariableNames', {'FeatureNames', 'DescriptionCol', 'DataType', 'Units'});

% Ensure additionalRows' FeatureNames column is compatible with metadataTable
if isa(metadataTable.FeatureNames, 'string')
    additionalRows.FeatureNames = string(additionalRows.FeatureNames);
end

% Join the two tables
metadataTable = [metadataTable; additionalRows];

% Write to Excel
writetable(metadataTable, metadataFile, 'Sheet', 'Metadata');
fprintf('SHAP metadata exported to: %s\n', metadataFile);

%% 2. Create SHAP_summary_all_targets.xlsx
fprintf('Creating SHAP_summary_all_targets.xlsx...\n');
summaryFile = fullfile(dataDir, 'SHAP_summary_all_targets.xlsx');

% Calculate feature importance (mean absolute SHAP value) for each target
featureImportance = zeros(numFeatures, numTargets);
for t = 1:numTargets
    featureImportance(:, t) = mean(abs(shapValues(:,:,t)), 1)';
end

% Create variable names for targets
targetColNames = cell(1, numTargets);
for t = 1:numTargets
    targetColNames{t} = ['Target_' num2str(t)];
end

% Create importance table for all targets
importanceTable = array2table(featureImportance, 'VariableNames', targetColNames);
importanceTable.FeatureName = featureNames';

% Calculate mean importance across all targets
meanImportance = mean(featureImportance, 2);
importanceTable.MeanImportance = meanImportance;

% Calculate relative importance (as percentage of max)
maxMeanImportance = max(meanImportance);
if maxMeanImportance > 0
    importanceTable.RelativeImportance = meanImportance / maxMeanImportance * 100;
else
    importanceTable.RelativeImportance = zeros(size(meanImportance));
end

% Sort by mean importance
[~, sortIdx] = sort(meanImportance, 'descend');
sortedImportanceTable = importanceTable(sortIdx, :);

% Add rank column
sortedImportanceTable.Rank = (1:numFeatures)';

% Reorder columns to put Rank first, then FeatureName
sortedImportanceTable = sortedImportanceTable(:, ...
    ['Rank', 'FeatureName', targetColNames, ...
     'MeanImportance', 'RelativeImportance']);

% Write to Excel
writetable(sortedImportanceTable, summaryFile, 'Sheet', 'ImportanceSummary');

% Add additional sheets with rankings for each target
for t = 1:numTargets
    % Sort features by importance for this target
    [sortedValues, sortIdx] = sort(featureImportance(:, t), 'descend');
    
    % Create a table for this target
    targetTable = table();
    targetTable.Rank = (1:numFeatures)';
    targetTable.FeatureName = featureNames(sortIdx)';
    targetTable.Importance = sortedValues;
    
    % Add relative importance (as percentage of max)
    maxImportance = max(sortedValues);
    if maxImportance > 0
        targetTable.RelativeImportance = sortedValues / maxImportance * 100;
    else
        targetTable.RelativeImportance = zeros(size(sortedValues));
    end
    
    % Add cumulative importance
    targetTable.CumulativeImportance = cumsum(targetTable.RelativeImportance);
    
    % Add significance flags (features above mean importance)
    meanTargetImportance = mean(sortedValues);
    targetTable.SignificantFeature = sortedValues > meanTargetImportance;
    
    % Write to Excel
    sheetName = sprintf('Target_%d', t);
    writetable(targetTable, summaryFile, 'Sheet', sheetName);
    fprintf('Exported ranking for Target %d\n', t);
end

% Add how-to-use sheet
howToUseTable = table();
howToUseTable.Section = {'Overview'; 'ImportanceSummary Sheet'; 'Target_X Sheets'; 'Interpreting Importance'};
howToUseTable.Description = {
    'This Excel workbook contains feature importance summaries from SHAP analysis across all targets.';
    'Shows overall feature importance ranking across all targets, with mean and relative importance metrics.';
    'Each target has its own sheet with detailed rankings, relative importance, and cumulative importance.';
    'Higher importance values indicate features that have greater impact on model predictions.'
};
writetable(howToUseTable, summaryFile, 'Sheet', 'HowToUse');

fprintf('SHAP summary exported to: %s\n', summaryFile);

%% 3. Export SHAP values for each target
disp('Exporting SHAP values to Excel files...');

for t = 1:numTargets
    % Create filename for this target
    filename = fullfile(dataDir, ['SHAP_values_Target ' num2str(t) '.xlsx']);
    
    % Extract SHAP values for this target
    targetShapValues = shapValues(:,:,t);
    
    % Create SHAP value column names
    shapColNames = cell(1, numFeatures);
    for i = 1:numFeatures
        shapColNames{i} = ['SHAP_' strrep(featureNames{i}, ' ', '_')];
    end
    
    % Create sample IDs
    sampleIDs = (1:numInstances)';
    
    % Create table with SHAP values
    shapTable = array2table(targetShapValues, 'VariableNames', shapColNames);
    shapTable.SampleID = sampleIDs;
    
    % Add input feature values if available
    if hasInputData
        % Create feature value column names
        valueColNames = cell(1, numFeatures);
        for i = 1:numFeatures
            valueColNames{i} = ['Value_' strrep(featureNames{i}, ' ', '_')];
            
            % Add feature values to table
            if size(input, 2) >= i
                shapTable.(valueColNames{i}) = input(:, i);
            else
                % Handle missing input dimensions
                shapTable.(valueColNames{i}) = NaN(numInstances, 1);
            end
        end
        
        % Reorder columns: SampleID, then alternating SHAP values and feature values
        allCols = {'SampleID'};
        for i = 1:numFeatures
            allCols{end+1} = shapColNames{i};
            allCols{end+1} = valueColNames{i};
        end
        
        % Check if columns exist and keep only valid ones
        existingCols = intersect(allCols, shapTable.Properties.VariableNames);
        shapTable = shapTable(:, existingCols);
    else
        % Just reorder to put SampleID first
        shapTable = shapTable(:, ['SampleID', shapColNames]);
    end
    
    % Write to Excel
    writetable(shapTable, filename, 'Sheet', 'SHAP_Values');
    
    % Add feature importance sheet
    [sortedValues, sortIdx] = sort(featureImportance(:, t), 'descend');
    importanceTable = table();
    importanceTable.Rank = (1:numFeatures)';
    importanceTable.FeatureName = featureNames(sortIdx)';
    importanceTable.Importance = sortedValues;
    importanceTable.RelativeImportance = sortedValues / max(sortedValues) * 100;
    
    writetable(importanceTable, filename, 'Sheet', 'FeatureImportance');
    
    % Add how-to-use sheet
    howToUseTable = table();
    howToUseTable.Section = {'Overview'; 'SHAP_Values Sheet'; 'FeatureImportance Sheet'; 'Interpreting SHAP Values'};
    howToUseTable.Description = {
        sprintf('This Excel workbook contains SHAP values for Target %d.', t);
        'Contains raw SHAP values for each sample and feature, along with the original feature values.';
        'Provides a ranking of features by importance for this specific target.';
        'Positive SHAP values indicate the feature pushed the prediction higher, negative values pushed it lower.'
    };
    writetable(howToUseTable, filename, 'Sheet', 'HowToUse');
    
    fprintf('SHAP values for Target %d exported to: %s\n', t, filename);
end

fprintf('All SHAP results successfully exported to Excel files in %s\n', dataDir); 