<<<<<<< HEAD
function run_shap_analysis(model_file, output_dir, shap_options)
% RUN_SHAP_ANALYSIS Run SHAP analysis on a trained neural network model
%
% INPUTS:
%   model_file - Path to the trained model .mat file
%   output_dir - Directory to save SHAP results
%   shap_options - (Optional) Structure with SHAP calculation parameters
%
% This function performs the full SHAP analysis workflow:
% 1. Load the trained model and preprocessing data
% 2. Calculate SHAP values for all targets
% 3. Generate visualization plots
% 4. Export SHAP values to Excel for further analysis
%
% Version 2.0.0 - Enhanced with improved SHAP implementation

    % Start timing
    start_time = tic;
    
    % Set default paths if not provided
    if nargin < 1 || isempty(model_file)
        % First look for best model from optimization
        best_model_path = fullfile('..', '..', 'results', 'best_model', 'best_model.mat');
        
        % If not found, use the full training model
        if (!exist(best_model_path, 'file'))
            best_model_path = fullfile('..', '..', 'results', 'training', 'trained_model.mat');
        end
        
        model_file = best_model_path;
    end
    
    % Default output directory (results/analysis/full/...)
    if nargin < 2 || isempty(output_dir)
        output_dir = fullfile('..', '..', 'results', 'analysis', 'full');
    end
    
    % Default SHAP options
    if nargin < 3
        shap_options = struct();
    end
    
    fprintf('Starting SHAP analysis workflow\n');
    fprintf('  Model file: %s\n', model_file);
    fprintf('  Output directory: %s\n', output_dir);
    
    % Create directories if they don't exist
    data_dir = fullfile(output_dir, 'data');
    fig_dir = fullfile(output_dir, 'figures');
    
    if ~exist(data_dir, 'dir')
        mkdir(data_dir);
        fprintf('Created directory: %s\n', data_dir);
    end
    
    if ~exist(fig_dir, 'dir')
        mkdir(fig_dir);
        fprintf('Created directory: %s\n', fig_dir);
    end
    
    % Load model
    fprintf('Loading model from %s\n', model_file);
    try
        model_data = load(model_file);
        
        % Check if the file contains the model and preprocessing info
        if ~isfield(model_data, 'model')
            error('Model file does not contain the expected ''model'' field');
        end
        
        model = model_data.model;
        
        % Load preprocessing information if available
        if isfield(model_data, 'preprocess_info')
            preprocess_info = model_data.preprocess_info;
        else
            preprocess_info = struct();
            warning('Preprocessing information not found in model file');
        end
        
    catch ME
        error('Failed to load model: %s', ME.message);
    end
    
    % Load input data
    fprintf('Loading input data\n');
    data_file = fullfile('..', 'model', 'CopyrolysisFeedstock.mat');
    try
        input_data = load(data_file);
        
        % Extract feature names if available
        if isfield(input_data, 'feature_names')
            feature_names = input_data.feature_names;
        else
            feature_names = cell(1, size(input_data.X, 2));
            for i = 1:length(feature_names)
                feature_names{i} = sprintf('Feature_%d', i);
            end
        end
        
        % Extract target names if available
        if isfield(input_data, 'target_names')
            target_names = input_data.target_names;
        else
            target_names = cell(1, size(input_data.Y, 2));
            for i = 1:length(target_names)
                target_names{i} = sprintf('Target %d', i);
            end
        end
        
        % Extract X and Y data
        X_data = input_data.X;
        Y_data = input_data.Y;
        
    catch ME
        error('Failed to load input data: %s', ME.message);
    end
    
    % Configure SHAP options
    if ~isfield(shap_options, 'num_background')
        shap_options.num_background = min(100, size(X_data, 1));
    end
    
    if ~isfield(shap_options, 'num_samples')
        shap_options.num_samples = size(X_data, 1);
    end
    
    if ~isfield(shap_options, 'background_data')
        shap_options.background_data = X_data;
    end
    
    if ~isfield(shap_options, 'num_permutations')
        shap_options.num_permutations = 100;
    end
    
    if ~isfield(shap_options, 'parallel')
        shap_options.parallel = true;
    end
    
    if ~isfield(shap_options, 'verbose')
        shap_options.verbose = true;
    end
    
    % Add path to calc_shap_values function
    addpath(fullfile('..', 'shap'));
    
    % Calculate SHAP values
    fprintf('Calculating SHAP values\n');
    try
        [shap_values, feature_names, X, target_names] = ...
            calc_shap_values(model, X_data, target_names, feature_names, shap_options);
    catch ME
        error('Error in SHAP calculation: %s', ME.message);
    end
    
    % Save SHAP results
    shap_results_file = fullfile(data_dir, 'shap_results.mat');
    fprintf('Saving SHAP results to %s\n', shap_results_file);
    save(shap_results_file, 'shap_values', 'feature_names', 'X', 'target_names', 'model');
    
    % Generate SHAP visualizations
    fprintf('Generating SHAP visualizations\n');
    try
        % Add path to visualization functions
        addpath(fullfile('..', 'visualization'));
        
        % Plot SHAP results (summary and beeswarm plots)
        plot_shap_results(shap_values, X, feature_names, target_names, fig_dir);
        
        % Create dependence plots for each feature and target
        create_dependence_plots(shap_values, X, feature_names, target_names, fig_dir);
    catch ME
        warning('Error generating visualizations: %s', ME.message);
        fprintf('Continuing with SHAP analysis workflow\n');
    end
    
    % Export SHAP values to Excel
    fprintf('Exporting SHAP values to Excel\n');
    try
        export_file = fullfile(output_dir, 'shap_analysis_results.xlsx');
        export_shap_to_excel(shap_values, X, feature_names, target_names, export_file);
        fprintf('Excel export complete: %s\n', export_file);
    catch ME
        warning('Error exporting to Excel: %s', ME.message);
    end
    
    % Create additional metadata Excel file
    try
        metadata_file = fullfile(data_dir, 'SHAP_metadata.xlsx');
        create_shap_metadata(shap_values, feature_names, target_names, metadata_file);
        fprintf('Metadata export complete: %s\n', metadata_file);
    catch ME
        warning('Error creating metadata file: %s', ME.message);
    end
    
    % Create summary Excel files for each target
    try
        for t = 1:length(target_names)
            summary_file = fullfile(data_dir, sprintf('SHAP_values_Target %d.xlsx', t));
            export_target_shap_values(shap_values{t}, X, feature_names, target_names{t}, summary_file);
        end
        
        % Create all targets summary file
        all_targets_file = fullfile(data_dir, 'SHAP_summary_all_targets.xlsx');
        export_all_targets_summary(shap_values, feature_names, target_names, all_targets_file);
        
    catch ME
        warning('Error creating summary files: %s', ME.message);
    end
    
    % Report completion and time
    fprintf('SHAP analysis workflow completed in %.2f seconds\n', toc(start_time));
    fprintf('Results saved to %s\n', output_dir);
end

function create_dependence_plots(shap_values, X, feature_names, target_names, fig_dir)
    % Create dependence plots for each feature and target
    
    % Set maximum number of features to plot (to avoid excessive plots)
    max_features = min(10, length(feature_names));
    
    % For each target
    for t = 1:length(shap_values)
        target_name = target_names{t};
        target_shap = shap_values{t};
        
        % Get feature importance for this target
        mean_abs_shap = mean(abs(target_shap), 1);
        [~, sorted_idx] = sort(mean_abs_shap, 'descend');
        
        % Plot top features
        for i = 1:max_features
            feature_idx = sorted_idx(i);
            feature_name = feature_names{feature_idx};
            
            % Clean feature name for filename
            clean_name = strrep(feature_name, ' ', '_');
            clean_name = strrep(clean_name, '/', '_');
            clean_name = strrep(clean_name, '\', '_');
            clean_name = strrep(clean_name, ':', '_');
            
            % Create figure
            h = figure('Visible', 'off');
            
            % Get this feature's SHAP values and the feature values
            feature_shap = target_shap(:, feature_idx);
            feature_values = X(:, feature_idx);
            
            % Plot dependence plot
            scatter(feature_values, feature_shap, 50, 'filled', 'MarkerFaceColor', 'b', 'MarkerFaceAlpha', 0.7);
            title(sprintf('Dependence plot for %s', feature_name), 'Interpreter', 'none');
            xlabel('Feature value', 'Interpreter', 'none');
            ylabel('SHAP value', 'Interpreter', 'none');
            grid on;
            
            % Add regression line
            hold on;
            p = polyfit(feature_values, feature_shap, 1);
            x_line = linspace(min(feature_values), max(feature_values), 100);
            y_line = polyval(p, x_line);
            plot(x_line, y_line, 'r-', 'LineWidth', 2);
            hold off;
            
            % Save figure
            fig_file = fullfile(fig_dir, sprintf('dependence_plot_%s.fig', clean_name));
            savefig(h, fig_file);
            
            % Save as PNG as well
            png_file = fullfile(fig_dir, sprintf('dependence_plot_%s.png', clean_name));
            print(h, png_file, '-dpng', '-r300');
            
            close(h);
        end
    end
end

function create_shap_metadata(shap_values, feature_names, target_names, file_path)
    % Create metadata Excel file with SHAP analysis information
    
    % Get feature importance for each target
    num_targets = length(shap_values);
    num_features = length(feature_names);
    
    % Create feature importance matrix
    feature_importance = zeros(num_features, num_targets);
    
    for t = 1:num_targets
        % Calculate mean absolute SHAP value for each feature
        feature_importance(:, t) = mean(abs(shap_values{t}), 1)';
    end
    
    % Create tables for Excel
    feature_table = table;
    feature_table.FeatureName = feature_names';
    
    for t = 1:num_targets
        col_name = ['Target_', strrep(target_names{t}, ' ', '_')];
        feature_table.(col_name) = feature_importance(:, t);
    end
    
    % Add overall importance (mean across targets)
    feature_table.Overall_Importance = mean(feature_importance, 2);
    
    % Sort by overall importance
    feature_table = sortrows(feature_table, 'Overall_Importance', 'descend');
    
    % Write to Excel
    writetable(feature_table, file_path, 'Sheet', 'Feature_Importance');
    
    % Add metadata sheet
    metadata = table;
    metadata.Property = {'Number of features'; 'Number of targets'; 'Analysis date'};
    metadata.Value = {num2str(num_features); num2str(num_targets); datestr(now)};
    
    writetable(metadata, file_path, 'Sheet', 'Metadata');
end

function export_target_shap_values(shap_values, X, feature_names, target_name, file_path)
    % Export SHAP values for a single target to Excel file
    
    % Create table for each sample's SHAP values
    samples_table = array2table(shap_values, 'VariableNames', feature_names);
    
    % Add actual feature values
    features_table = array2table(X, 'VariableNames', cellfun(@(x) ['Value_' x], feature_names, 'UniformOutput', false));
    
    % Combine tables
    combined_table = [samples_table features_table];
    
    % Add row numbers
    combined_table.Sample_ID = (1:size(shap_values, 1))';
    combined_table = combined_table(:, ['Sample_ID' combined_table.Properties.VariableNames(1:end-1)]);
    
    % Sort by sum of absolute SHAP values
    abs_sum = sum(abs(shap_values), 2);
    [~, sort_idx] = sort(abs_sum, 'descend');
    sorted_table = combined_table(sort_idx, :);
    
    % Write to Excel
    writetable(sorted_table, file_path, 'Sheet', 'Sample_SHAP_Values');
    
    % Write feature importance summary
    mean_abs_shap = mean(abs(shap_values), 1);
    [sorted_vals, sorted_idx] = sort(mean_abs_shap, 'descend');
    
    summary = table;
    summary.Rank = (1:length(feature_names))';
    summary.Feature = feature_names(sorted_idx)';
    summary.Importance = sorted_vals';
    summary.RelativeImportance = sorted_vals' / sum(sorted_vals);
    
    writetable(summary, file_path, 'Sheet', 'Feature_Importance');
end

function export_all_targets_summary(shap_values, feature_names, target_names, file_path)
    % Export summary of SHAP values for all targets
    
    % Create feature importance matrix
    num_features = length(feature_names);
    num_targets = length(target_names);
    
    importance_matrix = zeros(num_features, num_targets);
    
    % Calculate importance for each target
    for t = 1:num_targets
        importance_matrix(:, t) = mean(abs(shap_values{t}), 1)';
    end
    
    % Create summary table
    summary = table;
    summary.Feature = feature_names';
    
    for t = 1:num_targets
        col_name = ['Target_' strrep(target_names{t}, ' ', '_')];
        summary.(col_name) = importance_matrix(:, t);
    end
    
    % Add overall importance
    summary.Overall = mean(importance_matrix, 2);
    
    % Sort by overall importance
    summary = sortrows(summary, 'Overall', 'descend');
    
    % Write to Excel
    writetable(summary, file_path, 'Sheet', 'All_Targets_Summary');
    
    % Add rank table
    rank_table = table;
    rank_table.Feature = summary.Feature;
    
    for t = 1:num_targets
        col_name = ['Target_' strrep(target_names{t}, ' ', '_')];
        target_imp = summary.(col_name);
        [~, rank_idx] = sort(target_imp, 'descend');
        ranks = zeros(size(rank_idx));
        ranks(rank_idx) = 1:length(rank_idx);
        rank_table.(['Rank_' strrep(target_names{t}, ' ', '_')]) = ranks;
    end
    
    writetable(rank_table, file_path, 'Sheet', 'Feature_Ranks');
=======
function run_shap_analysis(model_file, output_dir, shap_options)
% RUN_SHAP_ANALYSIS Run SHAP analysis on a trained neural network model
%
% INPUTS:
%   model_file - Path to the trained model .mat file
%   output_dir - Directory to save SHAP results
%   shap_options - (Optional) Structure with SHAP calculation parameters
%
% This function performs the full SHAP analysis workflow:
% 1. Load the trained model and preprocessing data
% 2. Calculate SHAP values for all targets
% 3. Generate visualization plots
% 4. Export SHAP values to Excel for further analysis
%
% Version 2.0.0 - Enhanced with improved SHAP implementation

    % Start timing
    start_time = tic;
    
    % Set default paths if not provided
    if nargin < 1 || isempty(model_file)
        % First look for best model from optimization
        best_model_path = fullfile('..', '..', 'results', 'best_model', 'best_model.mat');
        
        % If not found, use the full training model
        if (!exist(best_model_path, 'file'))
            best_model_path = fullfile('..', '..', 'results', 'training', 'trained_model.mat');
        end
        
        model_file = best_model_path;
    end
    
    % Default output directory (results/analysis/full/...)
    if nargin < 2 || isempty(output_dir)
        output_dir = fullfile('..', '..', 'results', 'analysis', 'full');
    end
    
    % Default SHAP options
    if nargin < 3
        shap_options = struct();
    end
    
    fprintf('Starting SHAP analysis workflow\n');
    fprintf('  Model file: %s\n', model_file);
    fprintf('  Output directory: %s\n', output_dir);
    
    % Create directories if they don't exist
    data_dir = fullfile(output_dir, 'data');
    fig_dir = fullfile(output_dir, 'figures');
    
    if ~exist(data_dir, 'dir')
        mkdir(data_dir);
        fprintf('Created directory: %s\n', data_dir);
    end
    
    if ~exist(fig_dir, 'dir')
        mkdir(fig_dir);
        fprintf('Created directory: %s\n', fig_dir);
    end
    
    % Load model
    fprintf('Loading model from %s\n', model_file);
    try
        model_data = load(model_file);
        
        % Check if the file contains the model and preprocessing info
        if ~isfield(model_data, 'model')
            error('Model file does not contain the expected ''model'' field');
        end
        
        model = model_data.model;
        
        % Load preprocessing information if available
        if isfield(model_data, 'preprocess_info')
            preprocess_info = model_data.preprocess_info;
        else
            preprocess_info = struct();
            warning('Preprocessing information not found in model file');
        end
        
    catch ME
        error('Failed to load model: %s', ME.message);
    end
    
    % Load input data
    fprintf('Loading input data\n');
    data_file = fullfile('..', 'model', 'CopyrolysisFeedstock.mat');
    try
        input_data = load(data_file);
        
        % Extract feature names if available
        if isfield(input_data, 'feature_names')
            feature_names = input_data.feature_names;
        else
            feature_names = cell(1, size(input_data.X, 2));
            for i = 1:length(feature_names)
                feature_names{i} = sprintf('Feature_%d', i);
            end
        end
        
        % Extract target names if available
        if isfield(input_data, 'target_names')
            target_names = input_data.target_names;
        else
            target_names = cell(1, size(input_data.Y, 2));
            for i = 1:length(target_names)
                target_names{i} = sprintf('Target %d', i);
            end
        end
        
        % Extract X and Y data
        X_data = input_data.X;
        Y_data = input_data.Y;
        
    catch ME
        error('Failed to load input data: %s', ME.message);
    end
    
    % Configure SHAP options
    if ~isfield(shap_options, 'num_background')
        shap_options.num_background = min(100, size(X_data, 1));
    end
    
    if ~isfield(shap_options, 'num_samples')
        shap_options.num_samples = size(X_data, 1);
    end
    
    if ~isfield(shap_options, 'background_data')
        shap_options.background_data = X_data;
    end
    
    if ~isfield(shap_options, 'num_permutations')
        shap_options.num_permutations = 100;
    end
    
    if ~isfield(shap_options, 'parallel')
        shap_options.parallel = true;
    end
    
    if ~isfield(shap_options, 'verbose')
        shap_options.verbose = true;
    end
    
    % Add path to calc_shap_values function
    addpath(fullfile('..', 'shap'));
    
    % Calculate SHAP values
    fprintf('Calculating SHAP values\n');
    try
        [shap_values, feature_names, X, target_names] = ...
            calc_shap_values(model, X_data, target_names, feature_names, shap_options);
    catch ME
        error('Error in SHAP calculation: %s', ME.message);
    end
    
    % Save SHAP results
    shap_results_file = fullfile(data_dir, 'shap_results.mat');
    fprintf('Saving SHAP results to %s\n', shap_results_file);
    save(shap_results_file, 'shap_values', 'feature_names', 'X', 'target_names', 'model');
    
    % Generate SHAP visualizations
    fprintf('Generating SHAP visualizations\n');
    try
        % Add path to visualization functions
        addpath(fullfile('..', 'visualization'));
        
        % Plot SHAP results (summary and beeswarm plots)
        plot_shap_results(shap_values, X, feature_names, target_names, fig_dir);
        
        % Create dependence plots for each feature and target
        create_dependence_plots(shap_values, X, feature_names, target_names, fig_dir);
    catch ME
        warning('Error generating visualizations: %s', ME.message);
        fprintf('Continuing with SHAP analysis workflow\n');
    end
    
    % Export SHAP values to Excel
    fprintf('Exporting SHAP values to Excel\n');
    try
        export_file = fullfile(output_dir, 'shap_analysis_results.xlsx');
        export_shap_to_excel(shap_values, X, feature_names, target_names, export_file);
        fprintf('Excel export complete: %s\n', export_file);
    catch ME
        warning('Error exporting to Excel: %s', ME.message);
    end
    
    % Create additional metadata Excel file
    try
        metadata_file = fullfile(data_dir, 'SHAP_metadata.xlsx');
        create_shap_metadata(shap_values, feature_names, target_names, metadata_file);
        fprintf('Metadata export complete: %s\n', metadata_file);
    catch ME
        warning('Error creating metadata file: %s', ME.message);
    end
    
    % Create summary Excel files for each target
    try
        for t = 1:length(target_names)
            summary_file = fullfile(data_dir, sprintf('SHAP_values_Target %d.xlsx', t));
            export_target_shap_values(shap_values{t}, X, feature_names, target_names{t}, summary_file);
        end
        
        % Create all targets summary file
        all_targets_file = fullfile(data_dir, 'SHAP_summary_all_targets.xlsx');
        export_all_targets_summary(shap_values, feature_names, target_names, all_targets_file);
        
    catch ME
        warning('Error creating summary files: %s', ME.message);
    end
    
    % Report completion and time
    fprintf('SHAP analysis workflow completed in %.2f seconds\n', toc(start_time));
    fprintf('Results saved to %s\n', output_dir);
end

function create_dependence_plots(shap_values, X, feature_names, target_names, fig_dir)
    % Create dependence plots for each feature and target
    
    % Set maximum number of features to plot (to avoid excessive plots)
    max_features = min(10, length(feature_names));
    
    % For each target
    for t = 1:length(shap_values)
        target_name = target_names{t};
        target_shap = shap_values{t};
        
        % Get feature importance for this target
        mean_abs_shap = mean(abs(target_shap), 1);
        [~, sorted_idx] = sort(mean_abs_shap, 'descend');
        
        % Plot top features
        for i = 1:max_features
            feature_idx = sorted_idx(i);
            feature_name = feature_names{feature_idx};
            
            % Clean feature name for filename
            clean_name = strrep(feature_name, ' ', '_');
            clean_name = strrep(clean_name, '/', '_');
            clean_name = strrep(clean_name, '\', '_');
            clean_name = strrep(clean_name, ':', '_');
            
            % Create figure
            h = figure('Visible', 'off');
            
            % Get this feature's SHAP values and the feature values
            feature_shap = target_shap(:, feature_idx);
            feature_values = X(:, feature_idx);
            
            % Plot dependence plot
            scatter(feature_values, feature_shap, 50, 'filled', 'MarkerFaceColor', 'b', 'MarkerFaceAlpha', 0.7);
            title(sprintf('Dependence plot for %s', feature_name), 'Interpreter', 'none');
            xlabel('Feature value', 'Interpreter', 'none');
            ylabel('SHAP value', 'Interpreter', 'none');
            grid on;
            
            % Add regression line
            hold on;
            p = polyfit(feature_values, feature_shap, 1);
            x_line = linspace(min(feature_values), max(feature_values), 100);
            y_line = polyval(p, x_line);
            plot(x_line, y_line, 'r-', 'LineWidth', 2);
            hold off;
            
            % Save figure
            fig_file = fullfile(fig_dir, sprintf('dependence_plot_%s.fig', clean_name));
            savefig(h, fig_file);
            
            % Save as PNG as well
            png_file = fullfile(fig_dir, sprintf('dependence_plot_%s.png', clean_name));
            print(h, png_file, '-dpng', '-r300');
            
            close(h);
        end
    end
end

function create_shap_metadata(shap_values, feature_names, target_names, file_path)
    % Create metadata Excel file with SHAP analysis information
    
    % Get feature importance for each target
    num_targets = length(shap_values);
    num_features = length(feature_names);
    
    % Create feature importance matrix
    feature_importance = zeros(num_features, num_targets);
    
    for t = 1:num_targets
        % Calculate mean absolute SHAP value for each feature
        feature_importance(:, t) = mean(abs(shap_values{t}), 1)';
    end
    
    % Create tables for Excel
    feature_table = table;
    feature_table.FeatureName = feature_names';
    
    for t = 1:num_targets
        col_name = ['Target_', strrep(target_names{t}, ' ', '_')];
        feature_table.(col_name) = feature_importance(:, t);
    end
    
    % Add overall importance (mean across targets)
    feature_table.Overall_Importance = mean(feature_importance, 2);
    
    % Sort by overall importance
    feature_table = sortrows(feature_table, 'Overall_Importance', 'descend');
    
    % Write to Excel
    writetable(feature_table, file_path, 'Sheet', 'Feature_Importance');
    
    % Add metadata sheet
    metadata = table;
    metadata.Property = {'Number of features'; 'Number of targets'; 'Analysis date'};
    metadata.Value = {num2str(num_features); num2str(num_targets); datestr(now)};
    
    writetable(metadata, file_path, 'Sheet', 'Metadata');
end

function export_target_shap_values(shap_values, X, feature_names, target_name, file_path)
    % Export SHAP values for a single target to Excel file
    
    % Create table for each sample's SHAP values
    samples_table = array2table(shap_values, 'VariableNames', feature_names);
    
    % Add actual feature values
    features_table = array2table(X, 'VariableNames', cellfun(@(x) ['Value_' x], feature_names, 'UniformOutput', false));
    
    % Combine tables
    combined_table = [samples_table features_table];
    
    % Add row numbers
    combined_table.Sample_ID = (1:size(shap_values, 1))';
    combined_table = combined_table(:, ['Sample_ID' combined_table.Properties.VariableNames(1:end-1)]);
    
    % Sort by sum of absolute SHAP values
    abs_sum = sum(abs(shap_values), 2);
    [~, sort_idx] = sort(abs_sum, 'descend');
    sorted_table = combined_table(sort_idx, :);
    
    % Write to Excel
    writetable(sorted_table, file_path, 'Sheet', 'Sample_SHAP_Values');
    
    % Write feature importance summary
    mean_abs_shap = mean(abs(shap_values), 1);
    [sorted_vals, sorted_idx] = sort(mean_abs_shap, 'descend');
    
    summary = table;
    summary.Rank = (1:length(feature_names))';
    summary.Feature = feature_names(sorted_idx)';
    summary.Importance = sorted_vals';
    summary.RelativeImportance = sorted_vals' / sum(sorted_vals);
    
    writetable(summary, file_path, 'Sheet', 'Feature_Importance');
end

function export_all_targets_summary(shap_values, feature_names, target_names, file_path)
    % Export summary of SHAP values for all targets
    
    % Create feature importance matrix
    num_features = length(feature_names);
    num_targets = length(target_names);
    
    importance_matrix = zeros(num_features, num_targets);
    
    % Calculate importance for each target
    for t = 1:num_targets
        importance_matrix(:, t) = mean(abs(shap_values{t}), 1)';
    end
    
    % Create summary table
    summary = table;
    summary.Feature = feature_names';
    
    for t = 1:num_targets
        col_name = ['Target_' strrep(target_names{t}, ' ', '_')];
        summary.(col_name) = importance_matrix(:, t);
    end
    
    % Add overall importance
    summary.Overall = mean(importance_matrix, 2);
    
    % Sort by overall importance
    summary = sortrows(summary, 'Overall', 'descend');
    
    % Write to Excel
    writetable(summary, file_path, 'Sheet', 'All_Targets_Summary');
    
    % Add rank table
    rank_table = table;
    rank_table.Feature = summary.Feature;
    
    for t = 1:num_targets
        col_name = ['Target_' strrep(target_names{t}, ' ', '_')];
        target_imp = summary.(col_name);
        [~, rank_idx] = sort(target_imp, 'descend');
        ranks = zeros(size(rank_idx));
        ranks(rank_idx) = 1:length(rank_idx);
        rank_table.(['Rank_' strrep(target_names{t}, ' ', '_')]) = ranks;
    end
    
    writetable(rank_table, file_path, 'Sheet', 'Feature_Ranks');
>>>>>>> 5532f15296ed6babaee64d74dd966e573480e3cb
end