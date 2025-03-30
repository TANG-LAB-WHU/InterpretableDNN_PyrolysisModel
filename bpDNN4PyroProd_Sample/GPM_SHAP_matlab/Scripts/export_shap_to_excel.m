%% Export SHAP Results to Excel
% This script exports SHAP analysis results to Excel files for further analysis
% It creates multiple Excel files for different components of the SHAP analysis:
% 1. SHAP values for each sample and feature
% 2. Feature importance (mean absolute SHAP)
% 3. Base values and metadata
%
% All files are saved in the Data directory of the SHAP results folder

% Check if shapDir exists in the workspace
if ~exist('shapDir', 'var')
    % Get the script's directory
    scriptPath = mfilename('fullpath');
    [scriptDir, ~, ~] = fileparts(scriptPath);
    
    % Set default SHAP directory
    resultsDir = fullfile(fileparts(scriptDir), 'Results');
    shapDir = fullfile(resultsDir, 'SHAP_Analysis');
    
    warning('shapDir not found in workspace, using default path: %s', shapDir);
end

% Create data subfolder in shapDir if it doesn't exist
dataDir = fullfile(shapDir, 'Data');
if ~exist(dataDir, 'dir')
    mkdir(dataDir);
    disp(['Created data directory: ' dataDir]);
else
    disp(['Using existing data directory: ' dataDir]);
end

% Load SHAP results for export
disp('Processing SHAP values for Excel export...');

% Load SHAP data
shap_file = fullfile(dataDir, 'shap_results.mat');
if ~exist(shap_file, 'file')
    error('SHAP results file not found in %s. Run calc_shap_values.m first.', dataDir);
end

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
        targetNames = {'Target'};
    end
    disp('Using default target names (no targetNames found in SHAP results)');
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

% 1. Export SHAP values for each target
disp('Exporting SHAP values to Excel files...');

for t = 1:numTargets
    % Create filename for this target
    filename = fullfile(dataDir, ['SHAP_values_' strrep(targetNames{t}, '/', '_') '.xlsx']);
    
    % Create header row with feature names
    header = ['Sample ID', featureNames];
    
    % Extract SHAP values for this target
    targetShapValues = shapValues(:,:,t);
    
    % Create sample IDs
    sampleIDs = (1:numInstances)';
    
    % Combine into data for export
    exportData = [array2table(sampleIDs, 'VariableNames', {'Sample_ID'}), ...
                  array2table(targetShapValues, 'VariableNames', featureNames)];
    
    % Write to Excel
    try
        writetable(exportData, filename, 'Sheet', 'SHAP Values');
        fprintf('Exported SHAP values for %s to %s\n', targetNames{t}, filename);
    catch ME
        warning('ExportFailed:SHAPValues', '%s', ['Failed to export SHAP values for ' targetNames{t} ': ' ME.message]);
    end
    
    % 2. Export feature importance for this target
    % Calculate mean absolute SHAP values
    mean_abs_shap = mean(abs(targetShapValues), 1);
    
    % Sort features by importance
    [sorted_shap, sort_idx] = sort(mean_abs_shap, 'descend');
    sorted_names = featureNames(sort_idx);
    
    % Create feature importance table
    importanceTable = table(sorted_names', sorted_shap', ...
                           'VariableNames', {'Feature', 'Mean_Absolute_SHAP'});
    
    % Write to a second sheet in the same file
    try
        writetable(importanceTable, filename, 'Sheet', 'Feature Importance');
        fprintf('Exported feature importance for %s\n', targetNames{t});
    catch ME
        warning('ExportFailed:FeatureImportance', '%s', ['Failed to export feature importance for ' targetNames{t} ': ' ME.message]);
    end
    
    % 3. If input data is available, export raw feature values with SHAP values
    if hasInputData && size(input, 1) == numInstances
        try
            % Check if input needs transposing to match shapValues dimensions
            if size(input, 2) ~= numFeatures && size(input, 1) == numFeatures
                input = input';
            end
            
            % If input and SHAP dimensions match, create combined data
            if size(input, 2) >= numFeatures
                % Extract relevant columns if input has more columns than shapValues
                inputData = input(:, 1:numFeatures);
                
                % Create combined table with input values and SHAP values
                combinedTable = [array2table(sampleIDs, 'VariableNames', {'Sample_ID'})];
                
                % Add input values with 'Input_' prefix
                inputVarNames = strcat('Input_', featureNames);
                combinedTable = [combinedTable, array2table(inputData, 'VariableNames', inputVarNames)];
                
                % Add SHAP values with 'SHAP_' prefix
                shapVarNames = strcat('SHAP_', featureNames);
                combinedTable = [combinedTable, array2table(targetShapValues, 'VariableNames', shapVarNames)];
                
                % Write to a third sheet in the same file
                writetable(combinedTable, filename, 'Sheet', 'Combined Data');
                fprintf('Exported combined input and SHAP values for %s\n', targetNames{t});
            else
                warning('ExportLimited:DimensionMismatch', 'Input data dimensions do not match SHAP values. Skipping combined export.');
            end
        catch ME
            warning('ExportFailed:CombinedData', '%s', ['Failed to export combined data for ' targetNames{t} ': ' ME.message]);
        end
    end
end

% 4. Export base values and metadata
try
    % Create metadata table
    metaData = table(targetNames', baseValue', ...
                    'VariableNames', {'Target', 'Base_Value'});
    
    % Add additional metadata if available
    if isfield(results, 'modelInfo')
        % If there's model info, add it to the metadata
        metaData.Properties.Description = 'SHAP Analysis Metadata with Model Information';
    else
        metaData.Properties.Description = 'SHAP Analysis Metadata';
    end
    
    % Write metadata to a separate file
    metaFile = fullfile(dataDir, 'SHAP_metadata.xlsx');
    writetable(metaData, metaFile, 'Sheet', 'Base Values');
    fprintf('Exported base values and metadata to %s\n', metaFile);
    
    % If we have model info, add it to a second sheet
    if isfield(results, 'modelInfo')
        % This is a placeholder - in actual implementation, you would
        % format the model info appropriately for Excel
        writetable(struct2table(results.modelInfo), metaFile, 'Sheet', 'Model Info');
    end
catch ME
    warning('ExportFailed:Metadata', '%s', ['Failed to export metadata: ' ME.message]);
end

% 5. Export a comprehensive summary file with all targets
try
    % Create a summary Excel file with one sheet per target
    summaryFile = fullfile(dataDir, 'SHAP_summary_all_targets.xlsx');
    
    % First sheet: Base values for all targets
    baseValueTable = table(targetNames', baseValue', ...
                          'VariableNames', {'Target', 'Base_Value'});
    writetable(baseValueTable, summaryFile, 'Sheet', 'Base Values');
    
    % Second sheet: Feature importance across all targets
    % Calculate mean absolute SHAP for each feature and target
    importanceMatrix = zeros(numFeatures, numTargets);
    for t = 1:numTargets
        importanceMatrix(:, t) = mean(abs(shapValues(:,:,t)), 1)';
    end
    
    % Create overall importance - average across all targets
    overallImportance = mean(importanceMatrix, 2);
    
    % Sort by overall importance
    [~, sortIdx] = sort(overallImportance, 'descend');
    sortedFeatures = featureNames(sortIdx);
    sortedMatrix = importanceMatrix(sortIdx, :);
    
    % Create table with feature names and importance values per target
    importanceTable = array2table(sortedMatrix, 'VariableNames', targetNames);
    importanceTable = [table(sortedFeatures', 'VariableNames', {'Feature'}), importanceTable];
    
    % Add overall importance
    importanceTable.Overall = overallImportance(sortIdx);
    
    % Write to the summary file
    writetable(importanceTable, summaryFile, 'Sheet', 'Feature Importance');
    fprintf('Exported comprehensive summary to %s\n', summaryFile);
    
catch ME
    warning('ExportFailed:Summary', '%s', ['Failed to export comprehensive summary: ' ME.message]);
end

disp('SHAP data export to Excel completed!'); 