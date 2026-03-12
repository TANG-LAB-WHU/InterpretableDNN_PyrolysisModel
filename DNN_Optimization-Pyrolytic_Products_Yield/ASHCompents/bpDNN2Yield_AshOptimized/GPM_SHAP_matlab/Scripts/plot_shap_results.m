%% Plot SHAP Results
% This script generates visualizations for SHAP analysis results
%
% Features:
% - Creates multiple visualization types:
%   1. Bar plots showing top 20 most important features per target
%   2. Bar plots showing all features sorted by importance
%   3. Force plots showing top feature contributions for individual samples
%   4. Summary plots showing global feature importance patterns
% - All plots are automatically saved to the Figures directory
% - Figures are dynamically sized based on feature count
% - For plots with many features, font size and spacing are adjusted automatically
%
% Enhanced to provide comprehensive visualization of all features with adaptive layout

% Set default plot parameters for better visualization
set(0,'DefaultAxesFontSize',14);
set(0,'DefaultTextFontSize',14);
set(0,'DefaultAxesFontName','Arial');
set(0,'DefaultTextFontName','Arial');
set(0,'DefaultLineLineWidth',2);

% Define target full names with descriptions
targetFullNames = {'Target 1: Biochar', 'Target 2: Bioliquid', 'Target 3: Biogas'};

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

% Create data subfolder in shapDir if it doesn't exist
dataDir = fullfile(shapDir, 'Data');
if ~exist(dataDir, 'dir')
    mkdir(dataDir);
    disp(['Created data directory: ' dataDir]);
else
    disp(['Using existing data directory: ' dataDir]);
end

% Create figures subfolder in shapDir if it doesn't exist
figDir = fullfile(shapDir, 'Figures');
if ~exist(figDir, 'dir')
    mkdir(figDir);
    disp(['Created figures directory: ' figDir]);
else
    fprintf('Using existing figures directory: %s\n', figDir);
    % Check if directory is writable
    try
        % Use a safer approach to check directory permissions
        tempFile = fullfile(figDir, 'temp_write_test.txt');
        fileID = fopen(tempFile, 'w');
        if fileID == -1
            warning('The figures directory %s is not writable!', figDir);
        else
            fclose(fileID);
            delete(tempFile);
            disp('Figure directory is writable');
        end
    catch ME
        warning('FileCheck:Error', '%s', ['Could not check if figures directory is writable: ' ME.message]);
    end
end

% Load SHAP results for plotting
disp('Processing SHAP values for plotting...');

% Load SHAP data
shap_file = fullfile(dataDir, 'shap_results.mat');
if ~exist(shap_file, 'file')
    error('SHAP results file not found in %s. Run calc_shap_values.m first.', dataDir);
end

% Load with specific variable names to ensure they're all available
results = load(shap_file);
if ~isfield(results, 'shapValues') || ~isfield(results, 'baseValue') || ~isfield(results, 'input')
    error('Missing required variables in SHAP results. Re-run calc_shap_values.m.');
end

% Extract variables from the loaded structure
shapValues = results.shapValues;
baseValue = results.baseValue;
input = results.input;

% Setup variable names
if isfield(results, 'varNames')
    varNames = results.varNames;
    featureNames = varNames;
else
    % Create default feature names if not provided
    for i = 1:size(input, 2)
        featureNames{i} = sprintf('Feature %d', i);
    end
    disp('Using default feature names (no varNames found in SHAP results)');
end

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

% Ensure input dimensions are compatible with SHAP values
if size(input, 1) ~= numInstances || size(input, 2) < numFeatures
    fprintf('Input data dimensions (%d x %d) do not match SHAP values (%d x %d). ', ...
            size(input, 1), size(input, 2), numInstances, numFeatures);
    
    % Try to transpose if that would make dimensions match
    if size(input, 1) == numFeatures && size(input, 2) == numInstances
        fprintf('Transposing input to match SHAP values dimensions\n');
        input = input';
    elseif size(input, 2) < numFeatures
        % If input is smaller, pad it with NaN values
        input_padded = nan(size(input, 1), numFeatures);
        input_padded(:, 1:size(input, 2)) = input;
        input = input_padded;
        disp('Padded input data with NaN values to match SHAP dimensions');
    else
        warning('Some visualizations may be limited due to dimension mismatch.');
    end
end

% Plot global feature importance (mean absolute SHAP values)
for target = 1:numTargets
    % Calculate mean absolute SHAP values
    mean_abs_shap = mean(abs(shapValues(:,:,target)), 1);
    
    % Handle NaN values
    mean_abs_shap(isnan(mean_abs_shap)) = 0;
    
    % Sort features by importance
    [sorted_shap, sort_idx] = sort(mean_abs_shap, 'descend');
    
    % Limit to top 20 features for readability
    max_features_to_plot = min(20, length(sort_idx));
    
    % For each target, create a feature importance plot
    % Increase figure height to accommodate feature labels better
    figure('Position', [100, 100, 800, max(600, max_features_to_plot * 30)]);
    barh(sorted_shap(1:max_features_to_plot));
    
    % Ensure feature indices are valid
    valid_indices = sort_idx(1:max_features_to_plot);
    valid_indices = min(valid_indices, length(featureNames));
    
    % Get axis ytick labels
    yticks(1:max_features_to_plot);
    feature_labels = cell(max_features_to_plot, 1);
    for i = 1:max_features_to_plot
        feature_idx = valid_indices(i);
        feature_labels{i} = featureNames{feature_idx};
    end
    yticklabels(feature_labels);
    
    % Improve y-axis label display to avoid overlap
    ax = gca;
    ax.TickLabelInterpreter = 'none'; % Ensure special characters are displayed properly
    
    % Adjust axes position to provide more space for y-axis labels
    set(ax, 'Units', 'normalized');
    pos = get(ax, 'Position');
    
    % If we have many features, extend left margin to give more space for labels
    if max_features_to_plot > 6
        pos(1) = pos(1) + 0.05;
        pos(3) = pos(3) - 0.05;
        set(ax, 'Position', pos);
    end
    
    % Style the plot
    title(sprintf('Feature Importance (Top 20, %s)', targetFullNames{target}));
    xlabel('Mean |SHAP Value|');
    grid on;
    
    % Save plot with descriptive target name
    targetShortName = strrep(targetFullNames{target}, ' ', '');
    targetShortName = strrep(targetShortName, ':', '_');
    print(gcf, fullfile(figDir, sprintf('feature_importance_top20_%s.png', targetShortName)), '-dpng', '-r600');
    saveas(gcf, fullfile(figDir, sprintf('feature_importance_top20_%s.fig', targetShortName)));
    
    % Create new plot showing all features
    % Dynamically adjust the figure height based on the number of features
    figure_height = max(800, numFeatures * 25);
    figure('Position', [100, 100, 800, figure_height]);
    
    % Plot all features
    barh(sorted_shap);
    
    % Ensure all feature indices are valid
    all_valid_indices = sort_idx;
    all_valid_indices = min(all_valid_indices, length(featureNames));
    
    % Get axis ytick labels for all features
    yticks(1:numFeatures);
    all_feature_labels = cell(numFeatures, 1);
    for i = 1:numFeatures
        feature_idx = all_valid_indices(i);
        all_feature_labels{i} = featureNames{feature_idx};
    end
    yticklabels(all_feature_labels);
    
    % Adjust font size based on number of features
    if numFeatures > 30
        fontSize = max(8, 14 - (numFeatures - 30) * 0.15);
        set(gca, 'FontSize', fontSize);
    end
    
    % Improve y-axis label display to avoid overlap
    ax = gca;
    ax.TickLabelInterpreter = 'none'; % Ensure special characters are displayed properly
    
    % Adjust axes position to provide more space for y-axis labels
    set(ax, 'Units', 'normalized');
    pos = get(ax, 'Position');
    
    % For many features, extend left margin more for better label display
    margin_extension = min(0.2, 0.05 + (numFeatures - 6) * 0.005);
    pos(1) = pos(1) + margin_extension;
    pos(3) = pos(3) - margin_extension;
    set(ax, 'Position', pos);
    
    % Style the plot
    title(sprintf('Feature Importance (All Features, %s)', targetFullNames{target}));
    xlabel('Mean |SHAP Value|');
    grid on;
    
    % Save plot with descriptive target name
    print(gcf, fullfile(figDir, sprintf('feature_importance_all_%s.png', targetShortName)), '-dpng', '-r600');
    saveas(gcf, fullfile(figDir, sprintf('feature_importance_all_%s.fig', targetShortName)));
    
    % Close plots to free memory
    close all;
end

% Plot SHAP values for each instance (if predictions available)
if isfield(results, 'predictions') && ~isempty(results.predictions) && isnumeric(results.predictions)
    predictions = results.predictions;
    
    % Verify predictions dimensions
    if size(predictions, 1) ~= numTargets || size(predictions, 2) ~= numInstances
        warning(['Predictions dimensions (%d x %d) do not match expected dimensions (%d x %d). ' ...
                'Force plots may be incomplete.'], ...
                size(predictions, 1), size(predictions, 2), numTargets, numInstances);
    end
    
    % Plot only for first 5 samples
    num_samples_to_plot = min(5, numInstances);
    
    for idx = 1:num_samples_to_plot
        % Create plot for each target
        for target = 1:numTargets
            % Get SHAP values for this instance
            instance_shap = squeeze(shapValues(idx, :, target));
            
            % Skip if all NaN
            if all(isnan(instance_shap))
                continue;
            end
            
            % Sort by absolute values
            [~, abs_sort_idx] = sort(abs(instance_shap), 'descend');
            
            % Show top 10 features
            topFeatureCount = min(10, length(abs_sort_idx));
            topIdx = abs_sort_idx(1:topFeatureCount);
            
            % Get feature values
            feature_values = zeros(topFeatureCount, 1);
            for i = 1:topFeatureCount
                feature_idx = abs_sort_idx(i);
                if feature_idx <= size(input, 2)
                    feature_values(i) = input(idx, feature_idx);
                end
            end
            
            % Get feature names and SHAP values in sorted order
            sorted_shap = instance_shap(topIdx);
            sorted_names = featureNames(topIdx);
            
            % Create Python-style waterfall plot with descriptive target name
            createPythonStyleWaterfallPlot(baseValue(target), sorted_shap, sorted_names, feature_values, ...
                                          predictions(target, idx), idx, target, figDir, targetFullNames{target});
        end
    end
else
    warning('Variable ''predictions'' not found. Skipping individual force plots.');
end

% Helper function to create a Python SHAP style waterfall plot
function createPythonStyleWaterfallPlot(baseValue, shapValues, featureNames, featureValues, prediction, instanceIdx, targetIdx, figDir, targetName)
    % Number of features to display
    numFeatures = length(shapValues);
    
    % Create cumulative values for waterfall
    cumulativeValues = zeros(numFeatures + 1, 1);
    cumulativeValues(1) = baseValue;
    
    for i = 1:numFeatures
        cumulativeValues(i+1) = cumulativeValues(i) + shapValues(i);
    end
    
    % Create a new figure with good proportions
    figure('Position', [100, 100, 1200, max(600, numFeatures * 60)]);
    set(gcf, 'Color', 'white'); % Set background to white
    
    % Determine if we're displaying percentage values
    isPercentage = true; % Assume we're working with percentages
    
    % Format values to include percentage sign if appropriate
    if isPercentage
        baseValueStr = sprintf('%.2f%%', baseValue);
        predictionStr = sprintf('%.4f%%', prediction);
    else
        baseValueStr = sprintf('%.2f', baseValue);
        predictionStr = sprintf('%.4f', prediction);
    end
    
    % Format strings for feature display
    featureLabels = cell(numFeatures, 1);
    for i = 1:numFeatures
        featureName = featureNames{i};
        
        % Display all feature names in a consistent format keeping the original style
        % Keep underscores in the original feature names to match example images
        
        % Format the label with feature value
        featureLabels{i} = sprintf('%s = %.2f', featureName, featureValues(i));
    end
    
    % Start with bottom label for base value
    baseLabel = 'Base';
    
    % Plot data
    hold on;
    
    % Define positive and negative colors as in Python SHAP (red for positive, blue for negative)
    posColor = [0.94, 0.16, 0.16]; % Bright red for positive impact
    negColor = [0.16, 0.16, 0.94]; % Bright blue for negative impact
    
    % Plot base value as text on the left
    text(baseValue - 0.03 * range(cumulativeValues), 0, baseLabel, ...
         'HorizontalAlignment', 'right', 'VerticalAlignment', 'middle', 'FontWeight', 'bold', ...
         'FontSize', 12, 'Interpreter', 'none');
    
    % Draw grid lines first (so they appear behind the bars)
    % Horizontal grid lines
    for i = 0:numFeatures
        plot(xlim, [i, i], 'Color', [0.9, 0.9, 0.9], 'LineWidth', 0.5, 'LineStyle', '-');
    end
    
    % Vertical grid lines - determine reasonable intervals
    xRange = range(cumulativeValues);
    xMin = min(cumulativeValues) - 0.25 * xRange;
    xMax = max(cumulativeValues) + 0.15 * xRange;
    xStep = xRange / 10;
    for x = xMin:xStep:xMax
        plot([x, x], ylim, 'Color', [0.9, 0.9, 0.9], 'LineWidth', 0.5, 'LineStyle', '-');
    end
    
    % Plot feature contributions
    for i = 1:numFeatures
        % Determine position
        x_start = cumulativeValues(i);
        x_end = cumulativeValues(i+1);
        y_pos = i;
        
        % Determine color based on impact direction
        if shapValues(i) > 0
            color = posColor; % Red for positive impact
            edgeColor = [0.8, 0.1, 0.1]; % Slightly darker edge for positive
        else
            color = negColor; % Blue for negative impact
            edgeColor = [0.1, 0.1, 0.8]; % Slightly darker edge for negative
        end
        
        % Plot the bar with proper thickness and a subtle edge for better visibility
        barHeight = 0.7; % Control bar thickness
        
        % Add slight shadow for 3D effect (following Python SHAP style)
        if abs(shapValues(i)) > 0.01 * range(cumulativeValues)
            shadow = patch([x_start, x_end, x_end, x_start], ...
                          [y_pos-barHeight/2-0.03, y_pos-barHeight/2-0.03, y_pos+barHeight/2-0.03, y_pos+barHeight/2-0.03], ...
                          [0.2, 0.2, 0.2], 'EdgeColor', 'none', 'FaceAlpha', 0.2);
        end
        
        % Main bar
        bar = patch([x_start, x_end, x_end, x_start], ...
                    [y_pos-barHeight/2, y_pos-barHeight/2, y_pos+barHeight/2, y_pos+barHeight/2], ...
                    color, 'EdgeColor', edgeColor, 'LineWidth', 0.5);
        
        % Add feature value text on the left with better formatting
        text(min(cumulativeValues) - 0.07 * range(cumulativeValues), y_pos, featureLabels{i}, ...
             'HorizontalAlignment', 'right', 'VerticalAlignment', 'middle', 'FontSize', 12, 'Interpreter', 'none');
        
        % Add SHAP value as text near the bar if significant
        if abs(shapValues(i)) > 0.01 * range(cumulativeValues)
            textPos = (x_start + x_end) / 2;
            if shapValues(i) > 0
                textValue = sprintf('+%.2f', shapValues(i));
                if isPercentage
                    textValue = [textValue '%'];
                end
            else
                textValue = sprintf('%.2f', shapValues(i));
                if isPercentage
                    textValue = [textValue '%'];
                end
            end
            text(textPos, y_pos, textValue, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
                 'Color', [0, 0, 0], 'FontWeight', 'bold', 'FontSize', 11, 'Interpreter', 'none');
        end
    end
    
    % Connect the features with vertical lines
    for i = 1:numFeatures
        x_pos = cumulativeValues(i+1);
        plot([x_pos, x_pos], [i, i+1], 'k-', 'LineWidth', 0.7, 'Color', [0.5, 0.5, 0.5]);
    end
    
    % Add E[f(X)] at the bottom like in Python SHAP
    text(baseValue, -0.3, sprintf('E[f(X)] = %s', baseValueStr), ...
         'HorizontalAlignment', 'center', 'FontSize', 11, 'Color', [0.4, 0.4, 0.4], 'Interpreter', 'none');
    
    % Add f(x) at the top
    text(cumulativeValues(end), numFeatures + 0.3, sprintf('f(x) = %s', predictionStr), ...
         'HorizontalAlignment', 'center', 'FontSize', 11, 'Color', [0.4, 0.4, 0.4], 'Interpreter', 'none');
    
    % Add a thin vertical line for base value
    plot([baseValue, baseValue], [-0.1, 0.8], 'k-', 'LineWidth', 0.7, 'Color', [0.5, 0.5, 0.5]);
    
    % Add a thin vertical line for the final prediction
    plot([cumulativeValues(end), cumulativeValues(end)], [numFeatures, numFeatures + 0.3], 'k-', 'LineWidth', 0.7, 'Color', [0.5, 0.5, 0.5]);
    
    % Set up the plot
    xlim([min(cumulativeValues) - 0.3 * range(cumulativeValues), max(cumulativeValues) + 0.15 * range(cumulativeValues)]);
    ylim([-0.5, numFeatures + 0.8]);
    
    % Remove y-axis ticks and labels since we have custom labels
    set(gca, 'YTick', []);
    set(gca, 'YColor', 'none');
    
    % Format x-axis with subtle ticks
    ax = gca;
    set(ax, 'TickLength', [0.002, 0.002]);
    set(ax, 'XColor', [0.4, 0.4, 0.4]);
    set(ax, 'FontSize', 11);
    
    % Remove grid (we manually added better grid lines)
    grid off;
    
    % Remove box around plot
    box off;
    
    % Add a lighter horizontal line at y=0 for reference
    plot(xlim, [0, 0], 'Color', [0.7, 0.7, 0.7], 'LineWidth', 0.7, 'LineStyle', '-');
    
    % X-axis label
    xlabel('Prediction Value (%)', 'FontSize', 12, 'FontWeight', 'normal', 'Color', [0.3, 0.3, 0.3], 'Interpreter', 'none');
    
    % Add title with clear information
    % Extract product name from target name (e.g., "Biochar" from "Target 1: Biochar")
    [~, targetProduct] = strtok(targetName, ':');
    if ~isempty(targetProduct)
        targetProduct = strtrim(targetProduct(2:end)); % Remove colon and trim
    else
        targetProduct = targetName; % Fallback
    end
    
    % Use the target product name in the title
    title(sprintf('Individual Prediction Explanation (Instance %d, %s)', instanceIdx, targetName), ...
          'FontSize', 14, 'FontWeight', 'bold', 'Interpreter', 'none');
    
    % Add final prediction as large text at top
    text(min(cumulativeValues), numFeatures + 0.6, sprintf('Final Prediction: %s', predictionStr), ...
         'FontSize', 14, 'FontWeight', 'bold', 'HorizontalAlignment', 'left', 'Interpreter', 'none');
    
    % Save figure with high DPI using descriptive target name
    targetShortName = strrep(targetName, ' ', '');
    targetShortName = strrep(targetShortName, ':', '_');
    print(gcf, fullfile(figDir, sprintf('instance_explanation_%d_%s.png', instanceIdx, targetShortName)), '-dpng', '-r600');
    saveas(gcf, fullfile(figDir, sprintf('instance_explanation_%d_%s.fig', instanceIdx, targetShortName)));
    
    % Close the figure
    close(gcf);
end

fprintf('All SHAP visualizations saved to: %s\n', figDir); 