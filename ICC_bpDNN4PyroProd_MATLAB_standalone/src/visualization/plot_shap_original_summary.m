%% Original SHAP Summary Plot with Different Color Scheme
% This script creates the original version of the SHAP summary plot
% with a different colormap from the improved version

% Check if shapDir exists in the workspace
if ~exist('shapDir', 'var')
    % Get the script directory
    scriptPath = mfilename('fullpath');
    [scriptDir, ~, ~] = fileparts(scriptPath);
    
    % If not provided, set default paths based on the script location
    rootDir = fileparts(fileparts(scriptDir));
    
    % Try to determine which analysis mode we're in based on environment
    % If debug_run_analysis.m was the caller, use debug path
    % Otherwise use full analysis path
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
    else
        % Default to full analysis if called directly
        shapDir = fullfile(rootDir, 'results', 'analysis', 'full');
        fprintf('No caller detected. Using default path: %s\n', shapDir);
    end
end

% Create figures subfolder in shapDir if it doesn't exist
figDir = fullfile(shapDir, 'figures');
if ~exist(figDir, 'dir')
    mkdir(figDir);
    fprintf('Created figures directory: %s\n', figDir);
end

% Create data subfolder in shapDir if it doesn't exist
dataDir = fullfile(shapDir, 'data');
if ~exist(dataDir, 'dir')
    mkdir(dataDir);
    fprintf('Created data directory: %s\n', dataDir);
end

% Check if the shap values are already computed and available in the workspace
if ~exist('shapValues', 'var') || ~exist('input', 'var') || ~exist('featureNames', 'var')
    % Try to load from the data directory
    shapFile = fullfile(dataDir, 'shap_results.mat');
    
    if exist(shapFile, 'file')
        fprintf('Loading SHAP results from file: %s\n', shapFile);
        loadedData = load(shapFile);
        
        % Extract required variables from the loaded data
        if isfield(loadedData, 'shapValues')
            shapValues = loadedData.shapValues;
        else
            error('shapValues not found in loaded data');
        end
        
        if isfield(loadedData, 'input')
            input = loadedData.input;
        elseif isfield(loadedData, 'X')
            input = loadedData.X;
        else
            error('Input data not found in loaded data');
        end
        
        if isfield(loadedData, 'varNames')
            featureNames = loadedData.varNames;
        elseif isfield(loadedData, 'featureNames')
            featureNames = loadedData.featureNames;
        else
            error('Feature names not found in loaded data');
        end
    else
        error('SHAP results file not found: %s\nEither:\n1. Pass shapValues in workspace\n2. Run calc_shap_values.m first\n3. Specify correct shapDir', shapFile);
    end
end

% Function to format feature names for display
function formattedName = formatFeatureName(featureName, maxLength)
    % Ensure feature name is a string
    if ~ischar(featureName)
        featureName = char(featureName);
    end
    
    % Default maximum length is 25 characters
    if nargin < 2
        maxLength = 25;
    end
    
    % Truncate long feature names
    if length(featureName) > maxLength
        formattedName = [featureName(1:maxLength-3) '...'];
    else
        formattedName = featureName;
    end
    
    % Replace special characters to avoid display issues
    formattedName = strrep(formattedName, '_', ' ');
    formattedName = strrep(formattedName, '%', '%%');  % Prevent MATLAB interpreting as format specifier
end

% Prepare SHAP values - handle 3D arrays
if ndims(shapValues) == 3
    % For 3D shapValues (samples x features x targets), average across targets
    [numInstances, numFeatures, numTargets] = size(shapValues);
    fprintf('Found 3D SHAP values with %d instances, %d features, and %d targets\n', ...
            numInstances, numFeatures, numTargets);
    
    % Reshape to 2D by averaging across targets or taking first target
    shapValues2D = reshape(shapValues(:, :, 1), [numInstances, numFeatures]);
    shapValues = shapValues2D;
else
    [numInstances, numFeatures] = size(shapValues);
    fprintf('Found 2D SHAP values with %d instances and %d features\n', numInstances, numFeatures);
end

% Make sure feature names match number of features
if length(featureNames) ~= numFeatures
    fprintf('Warning: Feature names length (%d) does not match number of features (%d)\n', ...
            length(featureNames), numFeatures);
    if length(featureNames) < numFeatures
        % Extend feature names
        for i = length(featureNames)+1:numFeatures
            featureNames{i} = sprintf('Feature %d', i);
        end
    else
        % Truncate feature names
        featureNames = featureNames(1:numFeatures);
    end
end

% Calculate feature importance (mean absolute SHAP value)
featureImportance = mean(abs(shapValues), 1);
[sortedImportance, sortOrder] = sort(featureImportance, 'descend');

% Get number of top features to display
numTopFeatures = min(numFeatures, 20); % Display maximum of 20 features
topFeatures = sortOrder(1:numTopFeatures);

fprintf('Creating original SHAP summary plot with top %d features\n', numTopFeatures);

% Create figure for SHAP summary plot
figure('Position', [100, 100, 1000, max(600, numTopFeatures * 35)]);

% Set up plot area
ax = gca;
hold on;

% Define colors for plotting
colorNeg = [0, 0.6, 0.8];  % Cyan/blue for negative SHAP
colorPos = [0.8, 0.4, 0];  % Orange/red for positive SHAP

% Plot each feature's SHAP values
for i = 1:numTopFeatures
    featureIdx = topFeatures(i);
    
    % Safety check: ensure featureIdx is valid before accessing featureNames
    if featureIdx > length(featureNames) || featureIdx < 1
        warning('Feature index %d is out of bounds. Max index is %d.', featureIdx, length(featureNames));
        featureIdx = min(max(featureIdx, 1), length(featureNames));
    end
    
    % Get SHAP values for this feature
    featureShapValues = shapValues(:, featureIdx);
    
    % Get feature values for coloring (normalize to 0-1)
    featureValues = input(:, featureIdx);
    
    % Skip if all featureValues are NaN
    if all(isnan(featureValues))
        continue;
    end
    
    % Normalize feature values to [0,1] for coloring
    if max(featureValues) > min(featureValues)
        normalizedValues = (featureValues - min(featureValues)) / (max(featureValues) - min(featureValues));
    else
        normalizedValues = zeros(size(featureValues));
    end
    
    % Plot SHAP values as colored dots
    for j = 1:length(featureShapValues)
        % Skip NaN values
        if isnan(featureShapValues(j))
            continue;
        end
        
        % Determine color based on feature value
        if featureShapValues(j) >= 0
            % For positive SHAP values
            alpha = normalizedValues(j);
            color = colorPos * alpha + [1,1,1] * (1-alpha);
        else
            % For negative SHAP values
            alpha = normalizedValues(j);
            color = colorNeg * alpha + [1,1,1] * (1-alpha);
        end
        
        % Plot dot with jitter on y-axis
        jitter = (rand() - 0.5) * 0.2;
        scatter(featureShapValues(j), numTopFeatures - i + 1 + jitter, 30, color, 'filled', 'MarkerFaceAlpha', 0.7);
    end
    
    % Format and add feature name
    formattedName = formatFeatureName(featureNames{featureIdx});
    
    % Add feature name with improved formatting
    text(-max(abs(xlim)) * 1.05, numTopFeatures - i + 1, formattedName, ...
         'HorizontalAlignment', 'right', 'VerticalAlignment', 'middle', ...
         'FontSize', 10, 'FontWeight', 'normal', ...
         'Interpreter', 'none');
end

% Add a vertical line at x=0
plot([0, 0], [0, numTopFeatures + 1], 'k-', 'LineWidth', 1);

% Add horizontal grid lines
for i = 1:numTopFeatures
    plot(xlim, [i, i], 'k:', 'LineWidth', 0.5, 'Color', [0.8, 0.8, 0.8]);
end

% Format plot
title('SHAP Summary Plot (Feature Importance)', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('SHAP Value (impact on model output)', 'FontSize', 12, 'FontWeight', 'bold');
set(gca, 'YTick', []);
set(gca, 'Box', 'off');
set(gca, 'FontSize', 11);

% Create color legend
axes('Position', [0.75, 0.05, 0.2, 0.03]);
colormap_img = linspace(0, 1, 100);
[X, Y] = meshgrid(colormap_img, 1:10);
imagesc(X);
colormap(jet(100));
axis off;
text(0, -2, 'Low', 'HorizontalAlignment', 'left', 'FontSize', 9);
text(50, -2, 'Medium', 'HorizontalAlignment', 'center', 'FontSize', 9);
text(100, -2, 'High', 'HorizontalAlignment', 'right', 'FontSize', 9);
text(50, -10, 'Feature Value', 'HorizontalAlignment', 'center', 'FontSize', 9, 'FontWeight', 'bold');

% Save the summary plot (MATLAB .fig format for editing later)
saveas(gcf, fullfile(figDir, 'shap_summary.fig'));

% Save high-resolution PNG version
print(gcf, fullfile(figDir, 'shap_summary.png'), '-dpng', '-r600');

fprintf('Original SHAP summary plot saved to: %s\n', fullfile(figDir, 'shap_summary.png'));
fprintf('Original SHAP summary plot saved to: %s\n', fullfile(figDir, 'shap_summary.fig')); 