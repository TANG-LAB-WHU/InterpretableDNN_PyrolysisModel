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
    % Get the script's directory
    scriptPath = mfilename('fullpath');
    [scriptDir, ~, ~] = fileparts(scriptPath);
    
    % Set default SHAP directory
    resultsDir = fullfile(fileparts(scriptDir), 'Results');
    shapDir = fullfile(resultsDir, 'SHAP_Analysis_Debug');
    
    warning('shapDir not found in workspace, using default path: %s', shapDir);
end

% Create figures subfolder in shapDir if it doesn't exist
figDir = fullfile(shapDir, 'Figures');
if ~exist(figDir, 'dir')
    mkdir(figDir);
end

% Ensure the figure directory is valid and writable
fprintf('Using figure directory: %s\n', figDir);
if ~exist(figDir, 'dir')
    error('Figure directory does not exist after creation attempt: %s', figDir);
end

% Test if directory is writable by creating a small test file
testFilePath = fullfile(figDir, 'test_write.txt');
try
    fid = fopen(testFilePath, 'w');
    fprintf(fid, 'Test write access');
    fclose(fid);
    delete(testFilePath);
    fprintf('Figure directory is writable\n');
catch ME
    error('Figure directory is not writable: %s. Error: %s', figDir, ME.message);
end

% Check if the shap values are already computed and available in the workspace
if ~exist('shapValues', 'var') || ~exist('input', 'var') || ~exist('featureNames', 'var')
    % Try to load from the data directory
    dataDir = fullfile(shapDir, 'Data');
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
significantFeatureNames = featureNames(significantFeatureOrder);
numSignificantFeatures = length(significantFeatureIndices);

fprintf('Found %d significant features with importance above %.4f\n', ...
        numSignificantFeatures, significantFeatureThreshold);

%% FUNCTION: Create a beeswarm-style plot
function createBeeswarmPlot(shapValues, featureOrder, featureNames, numFeaturesToPlot, input, figTitle, figName, figDir)
    % Get dimensions
    [numInstances, ~] = size(shapValues);
    
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
    cmap = [
        linspace(0.8, 0, 256)', linspace(0, 0.8, 256)', linspace(0, 0.8, 256)'  % Red to Blue gradient
    ];
    
    % Define a colormap for feature values to match shap_summary.png
    cmap = jet(256);  % Standard jet colormap (blue=low, red=high)
    
    % Plot each feature's SHAP values
    for i = 1:numFeaturesToPlot
        featureIdx = featureOrder(i);
        
        % Check that featureIdx is within bounds
        if featureIdx > size(input, 2)
            warning('Feature index %d exceeds input dimensions %d. Skipping this feature.', ...
                   featureIdx, size(input, 2));
            continue;
        end
        
        % Get feature values and SHAP values
        featureVals = input(:, featureIdx);
        shapVals = shapValues(:, featureIdx);
        
        % Normalize feature values for coloring
        minVal = min(featureVals);
        maxVal = max(featureVals);
        normalizedFeatureVals = (featureVals - minVal) / (maxVal - minVal + eps);
        colorIndices = round(normalizedFeatureVals * 255) + 1;
        
        % Sort points by SHAP value for better visualization
        [sortedShapVals, sortIdx] = sort(shapVals);
        sortedColorIndices = colorIndices(sortIdx);
        
        % Create jitter for the y-position with improved non-overlapping algorithm
        yPositions = createJitterNonOverlapping(sortedShapVals, jitterAmount, i, featurePadding);
        
        % Plot scatter with color based on feature value
        scatter(sortedShapVals, yPositions, pointSize, cmap(sortedColorIndices, :), 'filled', 'MarkerFaceAlpha', pointAlpha);
    end
    
    % Add vertical line at x=0
    yLimits = [0.5, numFeaturesToPlot * featurePadding + 0.5];
    plot([0, 0], yLimits, 'k--', 'LineWidth', 1);
    
    % Customize plot
    yticks(1:featurePadding:numFeaturesToPlot*featurePadding);
    
    % For large feature sets, may need to use a smaller font size
    if numFeaturesToPlot > 30
        fontSize = max(8, 14 - (numFeaturesToPlot - 30) * 0.15);
        set(gca, 'FontSize', fontSize);
    end
    
    yticklabels(featureNames(1:numFeaturesToPlot));
    xlabel('SHAP value (impact on model output)');
    title(figTitle);
    grid on;
    
    % Improve y-axis label display to avoid overlap
    ax = gca;
    ax.TickLabelInterpreter = 'none'; % Ensure special characters are displayed properly
    
    % Adjust axes position to provide more space for y-axis labels
    set(ax, 'Units', 'normalized');
    pos = get(ax, 'Position');
    
    % If we have many features, extend left margin to give more space for labels
    if numFeaturesToPlot > 6
        margin_extension = min(0.2, 0.05 + (numFeaturesToPlot - 6) * 0.005);
        pos(1) = pos(1) + margin_extension;
        pos(3) = pos(3) - margin_extension;
        set(ax, 'Position', pos);
    end
    
    % Set x-axis limits symmetrically around zero for better visualization
    maxAbsX = max(abs(shapValues(:, featureOrder(1:numFeaturesToPlot))), [], 'all') * 1.1;
    xlim([-maxAbsX, maxAbsX]);
    
    % Set y-axis limits with proper padding to prevent cutoff
    ylim([0.5, numFeaturesToPlot * featurePadding + 0.5]);
    
    % Add colorbar to show feature value scale
    cb = colorbar;
    colormap(cmap);
    cb.Label.String = 'Feature value (normalized)';
    cb.Label.FontSize = 12;
    caxis([0 1]);
    cb.Ticks = [0, 0.25, 0.5, 0.75, 1];
    cb.TickLabels = {'Low', '', 'Medium', '', 'High'};

    % Add colorbar to show feature value scale to match shap_summary.png
    cb = colorbar;
    colormap(cmap);
    cb.Label.String = 'Feature value';
    cb.Label.FontSize = 12;
    cb.Label.FontWeight = 'bold';
    caxis([0 1]);
    cb.Ticks = [0, 0.5, 1];
    cb.TickLabels = {'Low', 'Medium', 'High'};
    
    % Save the plot as PNG and FIG
    saveas(gcf, fullfile(figDir, [figName '.png']));
    saveas(gcf, fullfile(figDir, [figName '.fig']));
    
    % Save the plot with high resolution PNG
    print(gcf, fullfile(figDir, [figName '.png']), '-dpng', '-r600');
    saveas(gcf, fullfile(figDir, [figName '.fig']));
    
    % Close figure to free up memory
    close(gcf);
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
    
    % Save the legend
    saveas(gcf, fullfile(figDir, 'color_scale_legend.png'));
    saveas(gcf, fullfile(figDir, 'color_scale_legend.fig'));

    % Save the legend with high resolution
    print(gcf, fullfile(figDir, 'color_scale_legend.png'), '-dpng', '-r600');
    saveas(gcf, fullfile(figDir, 'color_scale_legend.fig'));

    % Save the legend with high resolution
    print(gcf, fullfile(figDir, 'color_scale_legend.png'), '-dpng', '-r600');
    saveas(gcf, fullfile(figDir, 'color_scale_legend.fig'));
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