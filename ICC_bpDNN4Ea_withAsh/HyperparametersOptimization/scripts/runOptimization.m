% Main control script: Run hyperparameter optimization process

% Get script path
script_dir = fileparts(mfilename('fullpath'));
root_dir = fullfile(script_dir, '..');
results_dir = fullfile(root_dir, 'results');
best_model_dir = fullfile(root_dir, 'best_model');
data_dir = fullfile(root_dir, 'data');

% Create results directory
if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end

% Check if command line arguments exist
args = evalin('base', 'args');
if ~isempty(args)
    % If arguments exist, run the specified configuration
    config_id = str2double(args{1});
    fprintf('Running configuration ID: %d\n', config_id);
    
    % Update early stopping threshold if applicable
    try
        if mod(config_id, 5) == 0
            setEarlyStoppingThreshold(results_dir);
        end
    catch ME
        fprintf('Warning: Could not update early stopping threshold: %s\n', ME.message);
    end
    
    % Run optimization
    hyperParamOptimization(config_id, results_dir);
else
    % If no arguments, display configuration statistics and perform analysis
    countConfigurations();
    
    % Analyze results
    if exist(results_dir, 'dir') && ~isempty(dir(fullfile(results_dir, 'config_*')))
        fprintf('\nAnalyzing optimization results...\n');
        analyzeResults(results_dir);
        
        % Generate best model
        if ~exist(best_model_dir, 'dir')
            mkdir(best_model_dir);
        end
        fprintf('\nGenerating best model...\n');
        generateBestModel(results_dir, best_model_dir);
        
        % Verify models against baseline
        if exist(fullfile(data_dir, 'baseline_results.mat'), 'file')
            fprintf('\nVerifying models against baseline...\n');
            verifyModels(results_dir, data_dir);
        else
            fprintf('\nBaseline results not found, skipping model verification.\n');
        end
    else
        fprintf('\nResults directory is empty, cannot perform analysis. Please run optimization process first.\n');
    end
end
