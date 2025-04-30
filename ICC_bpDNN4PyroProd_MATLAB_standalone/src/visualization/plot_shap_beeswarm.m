%% SHAP Beeswarm Plot for Pyrolysis Model
% This script creates beeswarm-style plots for SHAP values
% Inspired by Python's SHAP beeswarm plot
% 
% This script can be run:
% 1. Standalone - it will try to find and load SHAP results automatically
% 2. As part of run_analysis.m or debug_run_analysis.m - it will use variables already in workspace
%
% Features:
% - Creates three types of plots:
%   1. Plot showing top 20 most important features
%   2. Plot showing only significant features (importance above mean)
%   3. Plot showing all features sorted by importance
% - Features are colored by their values (blue=low, red=high)
% - Points are jittered to create a beeswarm effect to show distribution density
% - Vertical line at x=0 shows the threshold between positive and negative impact
% - Automatically adjusts figure sizes and spacing based on feature count
%
% Enhanced for better visualization of complete feature set with dynamic scaling

% Set default plot parameters for better visualization
set(0,'DefaultAxesFontSize',14);
set(0,'DefaultTextFontSize',14);
set(0,'DefaultAxesFontName','Arial');
set(0,'DefaultTextFontName','Arial');
set(0,'DefaultLineLineWidth',2);

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

% Ensure the figure directory is valid and writable
fprintf('Using figure directory: %s\n', figDir);

% Check if the shap values are already computed and available in the workspace
if ~exist('shapValues', 'var') || ~exist('input', 'var') || ~exist('featureNames', 'var')
    % Try to load from the data directory
    shapFile = fullfile(dataDir, 'shap_results.mat');
    
    if exist(shapFile, 'file')
        fprintf('Loading SHAP results from file: %s\n', shapFile);
        loadedData = load(shapFile);
        
        % Check which variables are available in the loaded data
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
% For the "all features" plot, limit to top 20 for readability if there are too many
if numFeatures > 20
    numFeaturesToShow = 20;
    fprintf('Limiting the plot to top %d features for readability\n', numFeaturesToShow);
else
    numFeaturesToShow = numFeatures;
end

% For the significant features plot, use features whose importance is above the mean
% You can adjust this threshold to be more or less selective
significantFeatureThreshold = mean(featureImportance);
% Alternative: Use a more stringent threshold (mean + some fraction of std)
% significantFeatureThreshold = mean(featureImportance) + 0.5 * std(featureImportance);

significantFeatureIndices = find(featureImportance > significantFeatureThreshold);
[~, significantSortOrder] = sort(featureImportance(significantFeatureIndices), 'descend');
significantFeatureOrder = significantFeatureIndices(significantSortOrder);

% Safety check: make sure significantFeatureOrder values are within bounds of featureNames
invalidIndices = (significantFeatureOrder > length(featureNames) | significantFeatureOrder < 1);
if any(invalidIndices)
    warning('Found %d invalid indices in significantFeatureOrder. Filtering them out.', sum(invalidIndices));
    significantFeatureOrder = significantFeatureOrder(~invalidIndices);
    % Also rebuild significantFeatureNames array with only valid indices
    significantFeatureNames = cell(1, length(significantFeatureOrder));
    for i = 1:length(significantFeatureOrder)
        significantFeatureNames{i} = featureNames{significantFeatureOrder(i)};
    end
else
    % Only create significantFeatureNames if all indices are valid
    significantFeatureNames = featureNames(significantFeatureOrder);
end

numSignificantFeatures = length(significantFeatureOrder);

fprintf('Found %d significant features with importance above %.4f\n', ...
        numSignificantFeatures, significantFeatureThreshold);

%% Function to format feature names for display
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

%% FUNCTION: Create a beeswarm-style plot
function createBeeswarmPlot(shapValues, featureOrder, featureNames, numFeaturesToPlot, input, figTitle, figName, figDir)
    % Get dimensions
    [numInstances, numFeatures] = size(shapValues);
    
    % Safety check: ensure all feature indices are valid
    validIndices = (featureOrder <= numFeatures & featureOrder >= 1);
    if ~all(validIndices)
        warning('Filtering out %d invalid feature indices.', sum(~validIndices));
        featureOrder = featureOrder(validIndices);
        if iscell(featureNames)
            featureNames = featureNames(validIndices);
        end
        numFeaturesToPlot = min(numFeaturesToPlot, length(featureOrder));
        if numFeaturesToPlot == 0
            warning('No valid features to plot after filtering.');
            return;
        end
    end
    
    % Safety check: ensure numFeaturesToPlot doesn't exceed available features
    numFeaturesToPlot = min(numFeaturesToPlot, length(featureOrder));
    
    % Dynamically adjust spacing based on number of features to show
    if numFeaturesToPlot <= 20
        featurePadding = 1.0;  % Standard padding for fewer features
        height_multiplier = 80;  % Standard height multiplier
    else
        % For more features, reduce padding and increase height
        featurePadding = max(0.6, 1.0 - (numFeaturesToPlot - 20) * 0.01);  % Reduce padding gradually
        height_multiplier = max(60, 100 - (numFeaturesToPlot - 20) * 0.5);  % Adjust height multiplier
    end
    
    % Create figure with height proportional to number of features to avoid overlapping labels
    figure('Position', [100, 100, 1000, max(800, numFeaturesToPlot * height_multiplier)]);
    hold on;
    
    % Define parameters for jittering to create beeswarm effect
    jitterAmount = max(0.15, 0.3 - numFeaturesToPlot * 0.003);  % Reduce jittering for many features
    pointSize = max(10, 25 - numFeaturesToPlot * 0.2);  % Reduce point size for many features
    pointAlpha = 0.7;    % Transparency of points
    
    % Define a colormap for feature values (red = high, blue = low)
    % Use custom red-to-blue colormap to match color_scale_legend.png
    cmap = jet(256);
    
    % Plot features (in reverse order, so top feature is at the top)
    for i = 1:numFeaturesToPlot
        featureIdx = featureOrder(i);
        
        % Safety check: ensure featureIdx is valid before accessing featureNames
        if featureIdx > length(featureNames) || featureIdx < 1
            warning('Feature index %d is out of bounds. Max index is %d.', featureIdx, length(featureNames));
            featureIdx = min(max(featureIdx, 1), length(featureNames));
        end
        
        % Get SHAP values for this feature
        featureShap = shapValues(:, featureIdx);
        
        % Get feature values for coloring
        featureValues = input(:, featureIdx);
        
        % Normalize feature values to [0,1] for coloring
        if max(featureValues) > min(featureValues)
            normalizedValues = (featureValues - min(featureValues)) / (max(featureValues) - min(featureValues));
        else
            normalizedValues = zeros(size(featureValues));
        end
        
        % Plot each point with jitter and coloring
        for j = 1:numInstances
            if isnan(featureShap(j))
                continue;  % Skip NaN values
            end
            
            % Apply jitter based on SHAP value and add some randomness
            shap_val = featureShap(j);
            jitter = (rand() - 0.5) * jitterAmount * 2;  % Random jitter
            
            % Get color based on feature value (normalized)
            colorIdx = max(1, min(256, round(normalizedValues(j) * 255) + 1));
            color = cmap(colorIdx, :);
            
            % Plot point with transparency
            scatter(shap_val, numFeaturesToPlot-i+1+jitter, pointSize, color, 'filled', 'MarkerFaceAlpha', pointAlpha);
        end
        
        % Format and add feature name
        formattedName = formatFeatureName(featureNames{featureIdx});
        
        % Add feature name with improved font properties
        text(-max(abs(xlim)) * 1.05, numFeaturesToPlot-i+1, formattedName, ...
            'HorizontalAlignment', 'right', 'VerticalAlignment', 'middle', ...
            'FontSize', 10, 'FontWeight', 'normal', 'FontName', 'Arial', ...
            'Interpreter', 'none');  % 'none' to avoid interpreting '_' as subscript
    end
    
    % Add a vertical line at x=0
    xrange = max(abs(shapValues(:))) * 1.1;
    plot([0, 0], [0, numFeaturesToPlot + 1], 'k--', 'LineWidth', 1);
    
    % Set x-axis limits with some padding
    xlim([-xrange, xrange]);
    
    % Format axes
    xlabel('SHAP value (impact on model output)', 'FontSize', 12, 'FontWeight', 'bold');
    set(gca, 'YTick', []);  % Hide y-axis ticks
    set(gca, 'Box', 'off');  % Remove box
    set(gca, 'FontSize', 11);
    
    % Add title with better formatting
    title(figTitle, 'FontSize', 14, 'FontWeight', 'bold');
    
    % Save the plot (fig format for MATLAB editing and high-res PNG for reports)
    saveas(gcf, fullfile(figDir, [figName '.fig']));
    
    % Save high-resolution version (600 dpi)
    print(gcf, fullfile(figDir, [figName '.png']), '-dpng', '-r600');
end

% Improved helper function to create jittered y-positions that prevents overlap between features
function yPositions = createJitterNonOverlapping(xValues, jitterAmount, featureIndex, featurePadding)
    n = length(xValues);
    
    % Set the baseline position for this feature with consistent spacing
    basePosition = featureIndex * featurePadding;
    
    % Initialize all points at the baseline
    yPositions = ones(n, 1) * basePosition;
    
    % If no points or only one point, return baseline positions
    if n <= 1
        return;
    end
    
    % Define the maximum jitter as a fraction of the feature padding
    maxJitter = jitterAmount * 0.4 * featurePadding;
    
    % Use kernel density estimation to determine point density
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

% Original jitter function preserved for reference but no longer used
function yPositions = createJitter(xValues, jitterAmount, basePosition)
    n = length(xValues);
    yPositions = ones(n, 1) * basePosition;
    
    % Add jitter based on the density of x values
    % First, calculate a density estimate
    try
        [density, xGrid] = ksdensity(xValues);
        
        % Normalize density to the range [0, jitterAmount]
        normalizedDensity = density / max(density) * jitterAmount;
        
        % For each point, find the closest grid point and get its density
        for i = 1:n
            [~, idx] = min(abs(xGrid - xValues(i)));
            
            % Apply jitter with alternating direction based on point index
            % This creates a balanced "swarm" around the baseline
            if mod(i, 2) == 0
                yPositions(i) = basePosition + normalizedDensity(idx) * randn() * 0.5;
            else
                yPositions(i) = basePosition - normalizedDensity(idx) * randn() * 0.5;
            end
        end
    catch
        % Fallback if ksdensity fails
        rng(1); % For reproducible results
        yPositions = basePosition + jitterAmount * randn(n, 1) * 0.25;
    end
end

%% FUNCTION: Create a separate colorbar legend figure
function createColorbarLegend(figDir)
    % Create a simple figure just for the colorbar legend
    figure('Position', [100, 100, 600, 300]);
    
    % Create a gradient image
    x = linspace(0, 1, 256);
    y = ones(size(x));
    [X, Y] = meshgrid(x, y);
    
    % Plot the gradient image
    imagesc(X);
    colormap(jet(256)); % Standard jet colormap (blue=low, red=high)
    
    % Remove axes but keep colorbar
    axis off;
    
    % Add colorbar with explanatory labels
    cb = colorbar('Location', 'southoutside');
    cb.Label.String = 'Feature Value';
    cb.Label.FontSize = 14;
    cb.Label.FontWeight = 'bold';
    
    % Add explicit labels at min, mid, and max points
    cb.Ticks = [0, 0.5, 1];
    cb.TickLabels = {'Low', 'Medium', 'High'};
    
    % Add title explaining the color scheme
    title('Feature Value Color Scale', 'FontSize', 16);
    
    % Add explanatory text
    text(0.5, 0.7, {'In SHAP beeswarm plots:', ...
                    'Points represent individual samples', ...
                    'Color indicates the feature value (blue=low, red=high)', ...
                    'Position on x-axis shows impact on model output'}, ...
         'HorizontalAlignment', 'center', 'FontSize', 14);
    
    % Save the legend (only save once with high resolution)
    saveas(gcf, fullfile(figDir, 'color_scale_legend.fig'));
    print(gcf, fullfile(figDir, 'color_scale_legend.png'), '-dpng', '-r600');
    
    fprintf('Created standardized colorbar legend in: %s\n', figDir);
end

%% Create Beeswarm plots
disp('Creating SHAP beeswarm plots...');

% 1. Create plot for top features (most important ones)
fprintf('Creating beeswarm plot for top %d features...\n', min(numFeaturesToShow, numFeatures));
createBeeswarmPlot(shapValues, sortOrder, sortedFeatureNames, min(numFeaturesToShow, numFeatures), ...
                  input, 'SHAP Summary Plot (Top Features)', 'beeswarm_top_features', figDir);

% 2. Create plot for significant features (above mean importance)
if numSignificantFeatures > 0
    fprintf('Creating beeswarm plot for %d significant features...\n', numSignificantFeatures);
    createBeeswarmPlot(shapValues, significantFeatureOrder, significantFeatureNames, ...
                      numSignificantFeatures, input, 'SHAP Summary Plot (Significant Features)', ...
                      'beeswarm_significant_features', figDir);
end

% 3. New: Create plot for all features
fprintf('Creating beeswarm plot for all %d features...\n', numFeatures);
createBeeswarmPlot(shapValues, sortOrder, sortedFeatureNames, numFeatures, ...
                  input, 'SHAP Summary Plot (All Features)', 'beeswarm_all_features', figDir);

% Create colorbar legend figure
createColorbarLegend(figDir);

fprintf('SHAP beeswarm plots created and saved to %s\n', figDir); 