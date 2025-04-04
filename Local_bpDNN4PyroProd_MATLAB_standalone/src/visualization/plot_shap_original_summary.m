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
    resultsDir = fullfile(rootDir, 'results');
    
    % Try to determine which analysis mode we're in based on environment
    % If debug_run_analysis.m was the caller, use debug path
    % Otherwise use full analysis path
    dbstack_info = dbstack;
    if length(dbstack_info) > 1
        caller_name = dbstack_info(2).name;
        if contains(lower(caller_name), 'debug')
            shapDir = fullfile(resultsDir, 'analysis', 'debug');
            warning('No shapDir in workspace, using debug path: %s', shapDir);
        else
            shapDir = fullfile(resultsDir, 'analysis', 'full');
            warning('No shapDir in workspace, using full analysis path: %s', shapDir);
        end
    else
        % Default to full analysis if called directly
        shapDir = fullfile(resultsDir, 'analysis', 'full');
        warning('shapDir not found in workspace, using default path: %s', shapDir);
    end
end

% Create figures subfolder in shapDir if it doesn't exist
figDir = fullfile(shapDir, 'figures');
if ~exist(figDir, 'dir')
    mkdir(figDir);
end

% Set data directory
dataDir = fullfile(shapDir, 'data');

% Check if the shap values are already computed and available in the workspace
if ~exist('shapValues', 'var') || ~exist('input', 'var') || ~exist('featureNames', 'var')
    % Try to load from the data directory
    shapFile = fullfile(dataDir, 'shap_results.mat');
    
    if exist(shapFile, 'file')
        fprintf('Loading SHAP results from file: %s\n', shapFile);
        loadedData = load(shapFile);
        
        % Check which variables are available in the loaded data
        varsNeeded = {'shapValues', 'input', 'featureNames'};
        for i = 1:length(varsNeeded)
            if isfield(loadedData, varsNeeded{i})
                % Assign the variable to the base workspace
                assignin('base', varsNeeded{i}, loadedData.(varsNeeded{i}));
            else
                warning('Missing variable %s in loaded SHAP results', varsNeeded{i});
                
                % Check for alternative variable names
                if strcmp(varsNeeded{i}, 'featureNames') && isfield(loadedData, 'varNames')
                    assignin('base', 'featureNames', loadedData.varNames);
                    fprintf('Using varNames as featureNames\n');
                elseif strcmp(varsNeeded{i}, 'input') && isfield(loadedData, 'X_all')
                    assignin('base', 'input', loadedData.X_all');
                    fprintf('Using X_all as input\n');
                end
            end
        end
    else
        error('SHAP results file not found: %s\nPlease run calc_shap_values.m first', shapFile);
    end
end

% Make sure required variables are available
requiredVars = {'shapValues', 'input', 'featureNames'};
for i = 1:length(requiredVars)
    if ~exist(requiredVars{i}, 'var')
        error('Required variable ''%s'' not found. Cannot continue plotting.', requiredVars{i});
    end
end

% Prepare SHAP values - handle 3D arrays
if ndims(shapValues) == 3
    % For 3D shapValues (samples x features x targets), average across targets
    [numInstances, numFeatures, numTargets] = size(shapValues);
    fprintf('Found 3D SHAP values with %d instances, %d features, and %d targets\n', ...
            numInstances, numFeatures, numTargets);
    
    % Reshape to 2D by averaging across targets
    shapValues2D = zeros(numInstances, numFeatures);
    for i = 1:numInstances
        shapValues2D(i, :) = mean(squeeze(shapValues(i, :, :)), 2)';
    end
    shapValues = shapValues2D;
else
    [numInstances, numFeatures] = size(shapValues);
    fprintf('Found 2D SHAP values with %d instances and %d features\n', numInstances, numFeatures);
end

% Ensure input dimensions match shapValues dimensions
if size(input, 2) ~= numFeatures || size(input, 1) ~= numInstances
    fprintf('Input dimensions (%d x %d) do not match SHAP values dimensions (%d x %d)\n', ...
            size(input, 1), size(input, 2), numInstances, numFeatures);
    
    % Try to transpose if that would make dimensions match
    if size(input, 1) == numFeatures && size(input, 2) == numInstances
        fprintf('Transposing input to match SHAP values dimensions\n');
        input = input';
    elseif size(input, 1) == numFeatures
        fprintf('Resizing input to match number of instances in SHAP values\n');
        input = input(:, 1:numInstances)';
    else
        error('Cannot reconcile input dimensions with SHAP values dimensions');
    end
end

% Calculate feature importance (mean absolute SHAP value)
featureImportance = mean(abs(shapValues), 1);
[sortedImportance, sortOrder] = sort(featureImportance, 'descend');
sortedFeatureNames = featureNames(sortOrder);

% Determine the number of features to show in the plot
% Limit to top 10 features for a cleaner visualization
numFeaturesToShow = min(10, numFeatures);
fprintf('Creating original SHAP summary plot with top %d features\n', numFeaturesToShow);

%% Create the original SHAP summary plot
% Setup figure with clean style
% Increase figure height to accommodate feature labels better
figure('Position', [100, 100, 900, max(600, numFeaturesToShow * 60)]);
set(gcf, 'Color', 'white');
hold on;

% Set up the axes
ax = gca;
set(ax, 'FontSize', 12, 'Box', 'on', 'LineWidth', 1);
set(ax, 'YTick', 1:numFeaturesToShow, 'YTickLabel', {});
set(ax, 'TickLength', [0.005 0.005]);
grid on;

% Define a different colormap from improved - use blue to red color scheme
% Use standard jet colormap for contrast with improved version
originalCmap = jet(256);

% Helper function to create jittered y-positions for the beeswarm effect
function yPositions = createJitter(xValues, jitterAmount, basePosition)
    n = length(xValues);
    yPositions = ones(n, 1) * basePosition;
    
    % If no points or only one point, return baseline positions
    if n <= 1
        return;
    end
    
    % Define the maximum jitter as a fraction of the distance between features
    maxJitter = jitterAmount * 0.4;
    
    try
        % Group points by x-value proximity
        % Sort unique x values and find their counts
        [uniqueX, ~, ic] = unique(round(xValues * 100) / 100);
        counts = accumarray(ic, 1);
        
        % For points with the same x-value (or very close), distribute them vertically
        for i = 1:length(uniqueX)
            % Find indices of points with this x-value
            indices = find(abs(xValues - uniqueX(i)) < 0.01);
            count = length(indices);
            
            % Skip if only one point at this x-value
            if count <= 1
                continue;
            end
            
            % Calculate spacing to evenly distribute points within maxJitter
            spacing = (2 * maxJitter) / (count + 1);
            
            % Distribute points evenly from -maxJitter to +maxJitter
            for j = 1:count
                % Alternate between positive and negative jitter for better visual distribution
                if mod(j, 2) == 0
                    offset = (j / 2) * spacing;
                else
                    offset = -((j + 1) / 2) * spacing;
                end
                
                % Apply the jitter while ensuring it's within bounds
                yPositions(indices(j)) = basePosition + offset;
            end
        end
    catch
        % Fallback with simple random jitter if density estimation fails
        rng(1); % For reproducible results
        % Ensure jitter is constrained to prevent feature overlap
        yPositions = basePosition + (rand(n, 1) * 2 - 1) * maxJitter;
    end
    
    % Final check to ensure all points are within acceptable jitter range
    yPositions = min(max(yPositions, basePosition - maxJitter), basePosition + maxJitter);
end

% Sort features by importance and get the top N
topFeatures = sortOrder(1:numFeaturesToShow);
topFeatureNames = sortedFeatureNames(1:numFeaturesToShow);

% Define parameters for plotting
jitterAmount = 0.35;  % Controls the spread of points in the Y direction
pointSize = 25;      % Size of the dots
pointAlpha = 0.7;    % Transparency of points

% Plot each feature's SHAP values with beeswarm effect
for i = 1:numFeaturesToShow
    featureIdx = topFeatures(i);
    
    % Get feature values and SHAP values for this feature
    featureVals = input(:, featureIdx);
    shapVals = shapValues(:, featureIdx);
    
    % Normalize feature values for coloring (0 to 1 range)
    minVal = min(featureVals);
    maxVal = max(featureVals);
    normalizedFeatureVals = (featureVals - minVal) / (max(maxVal - minVal, eps));
    
    % Convert to color indices (1 to 256)
    colorIndices = max(1, min(256, round(normalizedFeatureVals * 255) + 1));
    
    % Sort points by SHAP value for better visual effect
    [sortedShapVals, sortIdx] = sort(shapVals);
    sortedColorIndices = colorIndices(sortIdx);
    
    % Create the jittered y-positions to create the beeswarm effect
    yPositions = createJitter(sortedShapVals, jitterAmount, i);
    
    % Plot dots with color based on feature value
    scatter(sortedShapVals, yPositions, pointSize, originalCmap(sortedColorIndices, :), 'filled', 'MarkerFaceAlpha', pointAlpha);
end

% Add a vertical line at x=0
plot([0, 0], [0.5, numFeaturesToShow+0.5], 'k--', 'LineWidth', 1);

% Set axis labels and title
xlabel('SHAP value (impact on model output)', 'FontSize', 14, 'FontWeight', 'bold');
title('Feature Impact on Model Output', 'FontSize', 16, 'FontWeight', 'bold');

% Add feature names on the y-axis
yTickLabels = cell(numFeaturesToShow, 1);
for i = 1:numFeaturesToShow
    yTickLabels{i} = topFeatureNames{i};
end
set(gca, 'YTick', 1:numFeaturesToShow, 'YTickLabels', yTickLabels);

% Improve y-axis label display to avoid overlap
ax = gca;
ax.TickLabelInterpreter = 'none'; % Ensure special characters are displayed properly
% Calculate proper spacing for labels
labelHeight = numFeaturesToShow * 0.9; % Adjust factor as needed
set(ax, 'Units', 'normalized');
pos = get(ax, 'Position');
% Extend left margin to give more space for labels
if numFeaturesToShow > 6
    pos(1) = pos(1) + 0.05;
    pos(3) = pos(3) - 0.05;
    set(ax, 'Position', pos);
end

% Add colorbar with custom colormap
colormap(originalCmap);
cb = colorbar;
ylabel(cb, 'Feature value', 'FontSize', 12);
caxis([0 1]);
cb.Ticks = [0, 0.5, 1];
cb.TickLabels = {'Low', 'Medium', 'High'};

% Set y-axis direction to show most important features at the top
set(gca, 'YDir', 'reverse');

% Set x-axis limits with a bit of padding for better visualization
maxAbsX = max(abs(shapValues(:, topFeatures)), [], 'all') * 1.05;
xlim([-maxAbsX, maxAbsX]);

% Set y-axis limits with a bit of padding
ylim([0.5, numFeaturesToShow+0.5]);

% Remove grid for a cleaner look
grid off;

% Save the original plot
hold off;
saveas(gcf, fullfile(figDir, 'shap_summary.png'));
saveas(gcf, fullfile(figDir, 'shap_summary.fig'));
fprintf('Original SHAP summary plot saved to: %s\n', fullfile(figDir, 'shap_summary.png'));

% Save the original plot
hold off;
print(gcf, fullfile(figDir, 'shap_summary.png'), '-dpng', '-r600');
saveas(gcf, fullfile(figDir, 'shap_summary.fig'));
fprintf('Original SHAP summary plot saved to: %s\n', fullfile(figDir, 'shap_summary.png')); 