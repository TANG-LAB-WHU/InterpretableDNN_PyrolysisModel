%% Plot SHAP Results
% This script creates visualizations of SHAP values computed by calc_shap_values.m
% It generates bar plots, force plots, and summary plots to help interpret
% the model predictions and feature importance.

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
    
    % Test if the directory is writable
    testFile = fullfile(figDir, 'test_write.txt');
    try
        fid = fopen(testFile, 'w');
        if fid == -1
            warning('Cannot write to figures directory: %s', figDir);
        else
            fprintf(fid, 'Test write access');
            fclose(fid);
            delete(testFile);
            fprintf('Figure directory is writable\n');
        end
    catch ME
        warning('Error testing write access to figures directory: %s\n%s', ...
                figDir, ME.message);
    end
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
        
        % Check for baseValue
        if isfield(loadedData, 'baseValue')
            baseValue = loadedData.baseValue;
        else
            warning('baseValue not found in loaded data, using default');
            baseValue = zeros(1, size(shapValues, 3));
        end
        
        % Store the loaded data for reference
        results = loadedData;
    else
        error('SHAP results file not found: %s\nEither:\n1. Pass shapValues in workspace\n2. Run calc_shap_values.m first\n3. Specify correct shapDir', shapFile);
    end
end

% Print information about the loaded data
fprintf('SHAP data loaded successfully\n');
fprintf('Shape of shapValues: %s\n', mat2str(size(shapValues)));
fprintf('Shape of input data: %s\n', mat2str(size(input)));
fprintf('Number of features: %d\n', length(featureNames));

% Prepare SHAP values for plotting
if ndims(shapValues) == 3
    % For 3D shapValues (samples x features x targets), we need to handle accordingly
    [numInstances, numFeatures, numTargets] = size(shapValues);
    fprintf('Found 3D SHAP values with %d instances, %d features, and %d targets\n', ...
        numInstances, numFeatures, numTargets);
    
    % For summary plots, average across targets
    shapValues2D = zeros(numInstances, numFeatures);
    for i = 1:numInstances
        shapValues2D(i, :) = mean(squeeze(shapValues(i, :, :)), 2)';
    end
else
    % Already 2D, just use directly
    [numInstances, numFeatures] = size(shapValues);
    fprintf('Found 2D SHAP values with %d instances and %d features\n', numInstances, numFeatures);
    shapValues2D = shapValues;
    numTargets = 1;
    % Reshape to make it 3D for consistent handling
    shapValues = reshape(shapValues, [numInstances, numFeatures, 1]);
end

%% Create feature importance plots for each target
fprintf('Creating feature importance plots for each target...\n');

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
    title(sprintf('Feature Importance (Top 20, Target %d)', target));
    xlabel('Mean |SHAP Value|');
    grid on;
    
    % Save plot
    print(gcf, fullfile(figDir, sprintf('feature_importance_top20_target_%d.png', target)), '-dpng', '-r600');
    saveas(gcf, fullfile(figDir, sprintf('feature_importance_top20_target_%d.fig', target)));
    
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
    title(sprintf('Feature Importance (All Features, Target %d)', target));
    xlabel('Mean |SHAP Value|');
    grid on;
    
    % Save plot
    print(gcf, fullfile(figDir, sprintf('feature_importance_all_target_%d.png', target)), '-dpng', '-r600');
    saveas(gcf, fullfile(figDir, sprintf('feature_importance_all_target_%d.fig', target)));
    
    % Close plots to free memory
    close all;
end

%% Create original force plots for individual instances
fprintf('Creating instance explanation plots...\n');

% Plot SHAP values for each instance per target
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
        
        % Create figure with height proportional to number of features
        figure('Position', [100, 100, 800, max(600, (topFeatureCount + 2) * 50)]);
        
        % Get SHAP values in sorted order
        sorted_shap = instance_shap(topIdx);
        
        % Setup waterfall plot values
        waterfall = zeros(topFeatureCount + 2, 1);
        waterfall(1) = baseValue(target);
        
        for i = 1:topFeatureCount
            waterfall(i+1) = waterfall(i) + sorted_shap(i);
        end
        waterfall(topFeatureCount+2) = waterfall(topFeatureCount+1);
        
        % Plot
        hold on;
        % Base value
        plot([1, 2], [waterfall(1), waterfall(1)], 'k-', 'LineWidth', 2);
        
        % Plot feature contributions
        for i = 1:topFeatureCount
            if sorted_shap(i) >= 0
                color = [0.2, 0.7, 0.3]; % Green for positive SHAP
            else
                color = [0.8, 0.3, 0.3]; % Red for negative SHAP
            end
            
            % Draw horizontal line
            plot([i+1, i+1], [waterfall(i), waterfall(i+1)], 'Color', color, 'LineWidth', 3);
            
            % Draw vertical line connecting to next feature
            plot([i+1, i+2], [waterfall(i+1), waterfall(i+1)], 'k-', 'LineWidth', 1);
        end
        
        % Final prediction
        plot([topFeatureCount+1, topFeatureCount+2], [waterfall(topFeatureCount+1), waterfall(topFeatureCount+1)], 'k-', 'LineWidth', 2);
        
        % Setup labels
        tick_labels = cell(topFeatureCount+2, 1);
        tick_labels{1} = 'Base value';
        
        for j = 1:topFeatureCount
            featureIdx = topIdx(j);
            % Ensure feature index is valid and within bounds
            if featureIdx <= size(input, 2) && featureIdx <= length(featureNames)
                % Safely check for valid input value
                if idx <= size(input, 1)
                    featVal = input(idx, featureIdx);
                    if isnan(featVal)
                        % Handle NaN values
                        tick_labels{j+1} = sprintf('%s = N/A', featureNames{featureIdx});
                    else
                        % Display the value
                        tick_labels{j+1} = sprintf('%s = %.2f', featureNames{featureIdx}, featVal);
                    end
                else
                    tick_labels{j+1} = sprintf('%s', featureNames{featureIdx});
                end
            else
                tick_labels{j+1} = sprintf('Feature %d', featureIdx);
            end
        end
        tick_labels{topFeatureCount+2} = 'Output value';
        
        % Customize plot
        xticks(1:topFeatureCount+2);
        set(gca, 'XTickLabel', tick_labels);
        xtickangle(45);
        grid on;
        title(sprintf('Instance %d Explanation (Target %d)', idx, target));
        xlabel('Features');
        ylabel('Output value');
        
        % Add value annotations
        for i = 1:topFeatureCount+2
            text(i, waterfall(min(i, topFeatureCount+1)), sprintf('%.3f', waterfall(min(i, topFeatureCount+1))), ...
                'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
        end
        
        % Add contribution labels
        for i = 1:topFeatureCount
            if abs(sorted_shap(i)) > 0.01
                ypos = (waterfall(i) + waterfall(i+1))/2;
                text(i+1, ypos, sprintf('%+.3f', sorted_shap(i)), ...
                    'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
                    'Color', [1 1 1], 'FontWeight', 'bold');
            end
        end
        
        % Save plot
        print(gcf, fullfile(figDir, sprintf('instance_explanation_%d_target_%d.png', idx, target)), '-dpng', '-r600');
        saveas(gcf, fullfile(figDir, sprintf('instance_explanation_%d_target_%d.fig', idx, target)));
    end
end

%% Create summary plot of feature importance
fprintf('Creating SHAP summary plot...\n');
featureImportance = mean(abs(shapValues2D), 1);
[sortedImportance, sortOrder] = sort(featureImportance, 'descend');

% Format feature names
formattedFeatureNames = cell(size(featureNames));
for i = 1:length(featureNames)
    formattedFeatureNames{i} = formatFeatureName(featureNames{i});
end

% Create a bar plot of feature importance
figure('Position', [100, 100, 800, 600]);
barh(sortedImportance(1:min(20, numFeatures)));
set(gca, 'YTick', 1:min(20, numFeatures), 'YTickLabel', formattedFeatureNames(sortOrder(1:min(20, numFeatures))));
set(gca, 'YDir', 'reverse');
xlabel('Mean |SHAP value|', 'FontSize', 12, 'FontWeight', 'bold');
title('Feature Importance Based on SHAP Values', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', 10);

% Save the plot
saveas(gcf, fullfile(figDir, 'feature_importance_bar.fig'));
print(gcf, fullfile(figDir, 'feature_importance_bar.png'), '-dpng', '-r600');
fprintf('Feature importance bar plot saved to: %s\n', fullfile(figDir, 'feature_importance_bar.png'));

%% Create a force plot for select instances
% We'll create force plots for the first few data points
fprintf('Creating SHAP force plots...\n');
numInstancesToShow = min(5, numInstances);

for i = 1:numInstancesToShow
    figure('Position', [100, 100, 1000, 300]);
    
    % Separate positive and negative contributions
    featureContribs = shapValues2D(i, :);
    posIndices = featureContribs > 0;
    negIndices = featureContribs < 0;
    
    % Create waterfall chart
    % Start with base value (average prediction)
    baseVal = mean(baseValue);
    
    % Plot base value
    bar(0, baseVal, 'FaceColor', [0.5, 0.5, 0.5]);
    hold on;
    
    % Plot positive contributions
    if any(posIndices)
        posFeatures = find(posIndices);
        [sortedPos, sortPosIdx] = sort(featureContribs(posIndices), 'descend');
        sortedPosFeatures = posFeatures(sortPosIdx);
        
        for j = 1:length(sortedPosFeatures)
            bar(j, featureContribs(sortedPosFeatures(j)), 'FaceColor', [0.2, 0.7, 0.3]);
            
            % Format feature name
            featureName = formatFeatureName(featureNames{sortedPosFeatures(j)}, 15);
            
            text(j, featureContribs(sortedPosFeatures(j))/2, ...
                 featureName, ...
                 'HorizontalAlignment', 'center', 'Rotation', 90, ...
                 'FontSize', 8);
        end
    end
    
    % Plot negative contributions
    if any(negIndices)
        negFeatures = find(negIndices);
        [sortedNeg, sortNegIdx] = sort(featureContribs(negIndices), 'ascend');
        sortedNegFeatures = negFeatures(sortNegIdx);
        
        offset = sum(posIndices);
        for j = 1:length(sortedNegFeatures)
            bar(j + offset, featureContribs(sortedNegFeatures(j)), 'FaceColor', [0.8, 0.3, 0.3]);
            
            % Format feature name
            featureName = formatFeatureName(featureNames{sortedNegFeatures(j)}, 15);
            
            text(j + offset, featureContribs(sortedNegFeatures(j))/2, ...
                 featureName, ...
                 'HorizontalAlignment', 'center', 'Rotation', 90, ...
                 'FontSize', 8);
        end
    end
    
    % Plot final prediction
    finalPred = baseVal + sum(featureContribs);
    bar(length(featureNames) + 1, finalPred, 'FaceColor', [0.2, 0.2, 0.7]);
    text(length(featureNames) + 1, finalPred/2, 'Prediction', ...
         'HorizontalAlignment', 'center', 'FontSize', 10);
    
    % Formatting
    xlim([-0.5, length(featureNames) + 1.5]);
    title(sprintf('SHAP Force Plot for Instance %d', i), 'FontSize', 12, 'FontWeight', 'bold');
    xlabel('Features', 'FontSize', 10);
    ylabel('SHAP Value Contribution', 'FontSize', 10);
    grid on;
    
    % Save the force plot
    saveas(gcf, fullfile(figDir, sprintf('force_plot_instance_%d.fig', i)));
    print(gcf, fullfile(figDir, sprintf('force_plot_instance_%d.png', i)), '-dpng', '-r600');
    fprintf('Force plot for instance %d saved to: %s\n', i, ...
            fullfile(figDir, sprintf('force_plot_instance_%d.png', i)));
end

%% Create dependence plots for top features
% These plots show how SHAP values depend on feature values
fprintf('Creating SHAP dependence plots...\n');
numTopFeatures = min(10, numFeatures);

% Check and fix dimensions of input data if needed
if size(input, 1) ~= numInstances
    % Input dimensions likely need to be transposed
    fprintf('Input dimensions (%d x %d) do not match SHAP values dimensions (%d x %d)\n', ...
        size(input, 1), size(input, 2), numInstances, numFeatures);
    
    if size(input, 2) == numInstances && size(input, 1) == numFeatures
        fprintf('Transposing input to match SHAP values dimensions\n');
        input = input';
    else
        fprintf('Warning: Cannot automatically fix input dimensions. Dependence plots may fail.\n');
        fprintf('Expected input size: %d x %d, Actual size: %d x %d\n', ...
            numInstances, numFeatures, size(input, 1), size(input, 2));
    end
end

for i = 1:numTopFeatures
    % Get the index of the current top feature
    featureIdx = sortOrder(i);
    
    % Safety check: ensure featureIdx is valid before accessing featureNames
    if featureIdx > length(featureNames) || featureIdx < 1
        warning('Feature index %d is out of bounds. Max index is %d.', featureIdx, length(featureNames));
        featureIdx = min(max(featureIdx, 1), length(featureNames));
    end
    
    figure('Position', [100, 100, 700, 500]);
    
    % Get feature values and SHAP values for this feature
    try
        featureVals = input(:, featureIdx);
        featureShapVals = shapValues2D(:, featureIdx);
        
        % Verify dimensions match before scatter plot
        if length(featureVals) ~= length(featureShapVals)
            warning('Feature values (%d) and SHAP values (%d) have different lengths for feature %s', ...
                length(featureVals), length(featureShapVals), featureNames{featureIdx});
            continue;
        end
        
        % Create scatter plot
        scatter(featureVals, featureShapVals, 40, 'filled', 'MarkerFaceAlpha', 0.7);
        
        % Add trend line
        hold on;
        try
            p = polyfit(featureVals, featureShapVals, 1);
            trendX = linspace(min(featureVals), max(featureVals), 100);
            trendY = polyval(p, trendX);
            plot(trendX, trendY, 'r-', 'LineWidth', 2);
        catch ME
            fprintf('Could not create trend line for feature %s: %s\n', ...
                featureNames{featureIdx}, ME.message);
        end
        
        % Add reference line at SHAP=0
        plot(xlim, [0, 0], 'k--');
        
        % Add labels and title
        xlabel(sprintf('Feature Value: %s', featureNames{featureIdx}));
        ylabel('SHAP Value');
        title(sprintf('SHAP Dependence Plot for %s', featureNames{featureIdx}));
        grid on;
        
        % Save the plot with consistent naming convention
        figName = sprintf('dependence_plot_%s', strrep(featureNames{featureIdx}, ' ', '_'));
        saveas(gcf, fullfile(figDir, [figName '.fig']));
        print(gcf, fullfile(figDir, [figName '.png']), '-dpng', '-r600');
        fprintf('Dependence plot for feature %s saved\n', featureNames{featureIdx});
    catch ME
        fprintf('Error creating dependence plot for feature %s: %s\n', ...
            featureNames{featureIdx}, ME.message);
    end
end

fprintf('SHAP plotting complete. All visualizations saved to: %s\n', figDir);

%% Helper functions
function formattedName = formatFeatureName(featureName)
    % This function formats feature names for better display in plots
    
    % Replace underscore with space
    formattedName = strrep(featureName, '_', ' ');
    
    % Handle special units in parentheses or after slash
    if contains(formattedName, '/%')
        % Replace /% with (%)
        formattedName = strrep(formattedName, '/%', ' (%)');
    end
    
    if contains(formattedName, '/Celsius')
        % Replace /Celsius with (°C)
        formattedName = strrep(formattedName, '/Celsius', ' (°C)');
    end
    
    if contains(formattedName, '/min')
        % Replace /min with (min)
        formattedName = strrep(formattedName, '/min', ' (min)');
    end
    
    if contains(formattedName, '/(K/min)')
        % Replace /(K/min) with (K/min)
        formattedName = strrep(formattedName, '/(K/min)', ' (K/min)');
    end
    
    % Capitalize first letter of each word
    words = strsplit(formattedName, ' ');
    for i = 1:length(words)
        if ~isempty(words{i})
            words{i}(1) = upper(words{i}(1));
        end
    end
    formattedName = strjoin(words, ' ');
end