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

% Create data subfolder in shapDir if it doesn't exist
dataDir = fullfile(shapDir, 'data');
if ~exist(dataDir, 'dir')
    mkdir(dataDir);
    disp(['Created data directory: ' dataDir]);
else
    disp(['Using existing data directory: ' dataDir]);
end

% Create figures subfolder in shapDir if it doesn't exist
figDir = fullfile(shapDir, 'figures');
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
            
            % Create figure with height proportional to number of features
            figure('Position', [100, 100, 800, max(600, (topFeatureCount + 2) * 50)]);
            
            % Get SHAP values in sorted order
            sorted_shap = instance_shap(topIdx);
            
            % Setup waterfall plot values
            waterfall = zeros(topFeatureCount + 2);
            waterfall(1) = baseValue(target);
            
            for i = 1:topFeatureCount
                waterfall(i+1) = waterfall(i) + sorted_shap(i);
            end
            waterfall(topFeatureCount+2) = waterfall(topFeatureCount+1);
            
            % Plot
            hold on;
            % Base value
            plot([waterfall(1), waterfall(1)], [1, 2], 'k-', 'LineWidth', 2);
            
            % Plot feature contributions
            for i = 1:topFeatureCount
                if sorted_shap(i) >= 0
                    color = 'b'; % Positive SHAP
                else
                    color = 'r'; % Negative SHAP
                end
                
                % Draw horizontal line
                plot([waterfall(i), waterfall(i+1)], [i+1, i+1], color, 'LineWidth', 2);
                
                % Draw vertical line
                plot([waterfall(i+1), waterfall(i+1)], [i+1, i+2], 'k-', 'LineWidth', 1);
            end
            
            % Final prediction
            plot([waterfall(topFeatureCount+1), waterfall(topFeatureCount+1)], ...
                [topFeatureCount+1, topFeatureCount+2], 'k-', 'LineWidth', 2);
            
            % Setup labels
            featureLabels = cell(topFeatureCount+1, 1);
            featureLabels{1} = 'Base';
            for j = 1:topFeatureCount
                featureIdx = topIdx(j);
                % Ensure feature index is valid and within bounds
                if featureIdx <= size(input, 2) && featureIdx <= length(featureNames)
                    % Safely check for valid input value
                    if idx <= size(input, 1)
                        featVal = input(idx, featureIdx);
                        if isnan(featVal)
                            % Handle NaN values
                            featureLabels{j+1} = sprintf('%s = N/A', featureNames{featureIdx});
                        else
                            % Display the value
                            featureLabels{j+1} = sprintf('%s = %.2f', featureNames{featureIdx}, featVal);
                        end
                    else
                        featureLabels{j+1} = sprintf('%s', featureNames{featureIdx});
                    end
                else
                    % Out of bounds
                    fprintf('Warning: Feature index %d is out of bounds (max features: %d, max names: %d)\n', ...
                        featureIdx, size(input, 2), length(featureNames));
                    safe_idx = min(featureIdx, length(featureNames));
                    if safe_idx > 0 && safe_idx <= length(featureNames)
                        featureLabels{j+1} = sprintf('%s = N/A', featureNames{safe_idx});
                    else
                        featureLabels{j+1} = sprintf('Feature %d = N/A', featureIdx);
                    end
                end
            end
            
            % Customize plot
            yticks(1:topFeatureCount+1);
            yticklabels(featureLabels);
            
            % Improve y-axis label display to avoid overlap
            ax = gca;
            ax.TickLabelInterpreter = 'none'; % Ensure special characters are displayed properly
            
            % Adjust axes position to provide more space for y-axis labels
            set(ax, 'Units', 'normalized');
            pos = get(ax, 'Position');
            
            % Extend left margin to give more space for labels
            if topFeatureCount > 5
                pos(1) = pos(1) + 0.08;
                pos(3) = pos(3) - 0.08;
                set(ax, 'Position', pos);
            end
            
            xlabel('Prediction Value');
            title(sprintf('Individual Prediction Explanation (Instance %d, Target %d)', idx, target));
            grid on;
            
            % Add prediction if available
            if size(predictions, 1) >= target && idx <= size(predictions, 2)
                try
                    pred_val = predictions(target, idx);
                    if ~isnan(pred_val)
                        text(waterfall(end), topFeatureCount+1.5, sprintf('Final Prediction: %.4f', pred_val), ...
                             'FontSize', 14, 'FontWeight', 'bold');
                    else
                        text(waterfall(end), topFeatureCount+1.5, 'Final Prediction: N/A', ...
                             'FontSize', 14, 'FontWeight', 'bold');
                    end
                catch
                    % Skip if prediction value is not accessible
                end
            end
            
            hold off;
            
            % Save figure
            print(gcf, fullfile(figDir, sprintf('instance_explanation_%d_target_%d.png', idx, target)), '-dpng', '-r600');
            saveas(gcf, fullfile(figDir, sprintf('instance_explanation_%d_target_%d.fig', idx, target)));
        end
    end
else
    warning('Variable ''predictions'' not found. Skipping individual force plots.');
end

fprintf('All SHAP visualizations saved to: %s\n', figDir); 