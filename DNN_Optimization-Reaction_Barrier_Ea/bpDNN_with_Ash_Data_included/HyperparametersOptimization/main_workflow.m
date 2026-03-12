% BP-DNN Hyperparameter Optimization Workflow
% This script orchestrates the entire workflow of BP-DNN hyperparameter optimization
% Starting from data preparation to final model selection
% The workflow uses matlab_src scripts as the first step and sequentially performs
% model hyperparameter optimization, saving figures similar to those in matlab_src/Figures

% Clear workspace
clc;
clear all;
close all;

fprintf('=====================================================================\n');
fprintf('  BP-DNN Hyperparameter Optimization Workflow\n');
fprintf('  %s\n', datestr(now));
fprintf('=====================================================================\n\n');

% Define paths
current_dir = pwd;
src_dir = fullfile(current_dir, 'matlab_src');
scripts_dir = fullfile(current_dir, 'scripts');
data_dir = fullfile(current_dir, 'data');
configs_dir = fullfile(current_dir, 'configs');
results_dir = fullfile(current_dir, 'results');
best_model_dir = fullfile(current_dir, 'best_model');
figures_dir = fullfile(current_dir, 'figures');

% Create directories if they don't exist
dirs_to_create = {data_dir, configs_dir, results_dir, best_model_dir, figures_dir};
for i = 1:length(dirs_to_create)
    if ~exist(dirs_to_create{i}, 'dir')
        mkdir(dirs_to_create{i});
        fprintf('Created directory: %s\n', dirs_to_create{i});
    end
end

% Add paths
addpath(src_dir);
addpath(scripts_dir);
addpath(data_dir);

% Copy required source files from matlab_src to data directory
fprintf('Copying required source files from matlab_src to data directory...\n');
required_files = {'CopyrolysisFeedstock.mat', 'RawInputData.xlsx', 'SampleIn_pred.mat', ...
    'SampleIn_scenario1.mat', 'SampleIn_scenario2.mat', 'SampleIn_scenario3.mat'};

for i = 1:length(required_files)
    src_file = fullfile(src_dir, required_files{i});
    if exist(src_file, 'file')
        copyfile(src_file, data_dir);
        fprintf('Copied %s to data directory\n', required_files{i});
    else
        fprintf('Warning: %s not found in matlab_src directory\n', required_files{i});
    end
end

% Step 1: Data Preparation
fprintf('\n--- Step 1: Data Preparation ---\n');
try
    run(fullfile(scripts_dir, 'prepare_data.m'));
    fprintf('Data preparation completed successfully.\n');
catch ME
    fprintf('Error during data preparation: %s\n', ME.message);
    return;
end

% Step 2: Generate Hyperparameter Configurations
fprintf('\n--- Step 2: Generate Hyperparameter Configurations ---\n');
try
    run(fullfile(scripts_dir, 'generate_configurations.m'));
    fprintf('Hyperparameter configurations generated successfully.\n');
catch ME
    fprintf('Error generating configurations: %s\n', ME.message);
    return;
end

% Step 3: Calculate Total Configurations
fprintf('\n--- Step 3: Calculate Total Configurations ---\n');
try
    countConfigurations();
catch ME
    fprintf('Error counting configurations: %s\n', ME.message);
    return;
end

% Step 4: Run Hyperparameter Optimization
fprintf('\n--- Step 4: Run Hyperparameter Optimization ---\n');
fprintf('Do you want to run the hyperparameter optimization? (y/n): ');
run_opt = input('', 's');

if strcmpi(run_opt, 'y')
    % Get total number of configurations
    config_count = getConfigCount();
    
    % Ask if running locally or on cluster
    fprintf('Run locally or submit to cluster? (local/cluster): ');
    run_mode = input('', 's');
    
    if strcmpi(run_mode, 'local')
        % Run locally (one by one or parallel)
        fprintf('Run in parallel? (y/n): ');
        parallel_opt = input('', 's');
        
        if strcmpi(parallel_opt, 'y')
            % Run in parallel using parfor
            run_parallel_optimization(config_count, results_dir);
        else
            % Run one by one with early stopping updates
            for config_id = 1:config_count
                fprintf('Running configuration %d of %d...\n', config_id, config_count);
                hyperParamOptimization(config_id, results_dir);
                
                % Update early stopping threshold every 5 configurations
                if mod(config_id, 5) == 0
                    try
                        setEarlyStoppingThreshold(results_dir);
                    catch ME
                        fprintf('Warning: Could not update early stopping threshold: %s\n', ME.message);
                    end
                end
            end
        end
    else
        % Generate submission script for cluster
        fprintf('Generating submission script for cluster...\n');
        generate_submission_script(config_count);
        fprintf('Please use the generated hyperParamOptimization.sh script to submit jobs to the cluster.\n');
    end
else
    fprintf('Skipping hyperparameter optimization run.\n');
end

% Step 5: Analyze Results and Select Best Model
fprintf('\n--- Step 5: Analyze Results and Select Best Model ---\n');
fprintf('Do you want to analyze results and generate the best model? (y/n): ');
analyze_opt = input('', 's');

if strcmpi(analyze_opt, 'y')
    % Check if results directory contains results
    if ~isempty(dir(fullfile(results_dir, 'config_*')))
        fprintf('Analyzing results...\n');
        analyzeResults(results_dir);
        
        fprintf('Generating best model...\n');
        generateBestModel(results_dir, best_model_dir);
        
        fprintf('Analysis and best model generation completed.\n');
    else
        fprintf('No results found to analyze. Run hyperparameter optimization first.\n');
    end
else
    fprintf('Skipping results analysis.\n');
end

% Step 6: Visualize Results
fprintf('\n--- Step 6: Visualize Results ---\n');
fprintf('Do you want to visualize results? (y/n): ');
viz_opt = input('', 's');

if strcmpi(viz_opt, 'y')
    % Check if results have been analyzed
    if exist(fullfile(results_dir, 'all_results_sorted.csv'), 'file')
        fprintf('Generating visualizations...\n');
        visualize_results(results_dir, figures_dir);
        fprintf('Visualizations completed and saved to %s.\n', figures_dir);
    else
        fprintf('No analyzed results found. Run analysis first.\n');
    end
else
    fprintf('Skipping results visualization.\n');
end

% Step 7: Verify Models Against Baseline
fprintf('\n--- Step 7: Verify Models Against Baseline ---\n');
fprintf('Do you want to verify optimized models against baseline? (y/n): ');
verify_opt = input('', 's');

if strcmpi(verify_opt, 'y')    % Check if best model has been generated
    if exist(fullfile(best_model_dir, 'Results_trained.mat'), 'file') && ...
       exist(fullfile(data_dir, 'baseline_results.mat'), 'file')
        fprintf('Verifying models against baseline...\n');
        verifyModels(results_dir, data_dir);
        fprintf('Model verification completed.\n');
    else
        fprintf('Required files not found. Make sure baseline and best models exist.\n');
    end
else
    fprintf('Skipping model verification.\n');
end

% End of workflow
fprintf('\n=====================================================================\n');
fprintf('  BP-DNN Hyperparameter Optimization Workflow Completed\n');
fprintf('  %s\n', datestr(now));
fprintf('=====================================================================\n');

% Helper function to get config count
function count = getConfigCount()
    % Load hyperparameter configuration
    config_file = fullfile(pwd, 'hyperparameters.json');
    fid = fopen(config_file, 'r');
    raw_text = fread(fid, '*char')';
    fclose(fid);
    config = jsondecode(raw_text);
    
    % Calculate all possible combinations
    num_lr = length(config.learning_rate);
    num_momentum = length(config.momentum);
    num_hidden_layers = length(config.hidden_layers);
    num_transfer_funcs = length(config.transfer_functions);
    num_lr_inc = length(config.lr_increase_factor);
    num_lr_dec = length(config.lr_decrease_factor);
    num_div_ratio = size(config.division_ratio, 1);
    num_max_fail = length(config.max_fail);
    num_max_epochs = length(config.max_epochs);
    
    count = num_lr * num_momentum * num_hidden_layers * ...
             num_transfer_funcs * num_lr_inc * num_lr_dec * ...
             num_div_ratio * num_max_fail * num_max_epochs;
end

% Helper function to run parallel optimization
function run_parallel_optimization(config_count, results_dir)
    % Create parallel pool
    if isempty(gcp('nocreate'))
        pool = parpool();
    end
    
    % Run optimization in parallel
    parfor config_id = 1:config_count
        fprintf('Running configuration %d of %d...\n', config_id, config_count);
        try
            hyperParamOptimization(config_id, results_dir);
        catch ME
            fprintf('Error running configuration %d: %s\n', config_id, ME.message);
        end
    end
end

% Helper function to generate submission script
function generate_submission_script(config_count)
    script_file = fullfile(pwd, 'hyperParamOptimization.sh');
    
    % Create job script
    fid = fopen(script_file, 'w');
      % Write script header
    fprintf(fid, '#!/bin/bash\n');
    fprintf(fid, '#\n');
    fprintf(fid, '# Hyperparameter Optimization Script for BP-DNN\n');
    fprintf(fid, '# For use on NCSA ICC UIUC supercomputing platform\n');
    fprintf(fid, '# Updated for SLURM job scheduler\n');
    fprintf(fid, '#\n');
    fprintf(fid, '# This script combines:\n');
    fprintf(fid, '# 1. Parameter configuration and job submission\n');
    fprintf(fid, '# 2. Individual job execution\n');
    fprintf(fid, '# 3. Result analysis\n');
    fprintf(fid, '#\n\n');
    
    % Usage information
    fprintf(fid, '# Function to print usage information\n');
    fprintf(fid, 'print_usage() {\n');
    fprintf(fid, '    echo "Usage: $0 [COMMAND]"\n');
    fprintf(fid, '    echo ""\n');
    fprintf(fid, '    echo "Commands:"\n');
    fprintf(fid, '    echo "  submit      Calculate configurations and submit array jobs"\n');
    fprintf(fid, '    echo "  analyze     Analyze results after all jobs complete"\n');
    fprintf(fid, '    echo "  help        Display this help message"\n');
    fprintf(fid, '    echo ""\n');
    fprintf(fid, '    echo "Example:"\n');
    fprintf(fid, '    echo "  $0 submit   # Submit all hyperparameter optimization jobs"\n');
    fprintf(fid, '    echo "  $0 analyze  # Analyze results after completion"\n');
    fprintf(fid, '}\n\n');
    
    % Submit jobs function
    fprintf(fid, '# Function to submit jobs\n');
    fprintf(fid, 'submit_jobs() {\n');
    fprintf(fid, '    echo "======================================================================"\n');
    fprintf(fid, '    echo "  Hyperparameter Optimization Job Submission"\n');
    fprintf(fid, '    echo "  $(date)"\n');
    fprintf(fid, '    echo "======================================================================"\n');
    fprintf(fid, '    \n');
    fprintf(fid, '    # Set array job range\n');
    fprintf(fid, '    ARRAY_RANGE="1-%d"\n', config_count);
    fprintf(fid, '    \n');
    fprintf(fid, '    # Create job script\n');
    fprintf(fid, '    JOB_SCRIPT="hyperopt_job_$$.sh"\n');
    fprintf(fid, '    cat > $JOB_SCRIPT << EOL\n');    fprintf(fid, '#!/bin/bash\n');
    fprintf(fid, '#SBATCH --job-name=bp_dnn_hyperparam_opt\n');
    fprintf(fid, '#SBATCH --nodes=1\n');
    fprintf(fid, '#SBATCH --ntasks-per-node=1\n');
    fprintf(fid, '#SBATCH --time=48:00:00\n');
    fprintf(fid, '#SBATCH --output=%%j.out\n');
    fprintf(fid, '#SBATCH --error=%%j.err\n');
    fprintf(fid, '#SBATCH --account=siqi-ic\n');
    fprintf(fid, '#SBATCH --partition=IllinoisComputes\n');
    fprintf(fid, '\n');
    fprintf(fid, '# Set working directory\n');
    fprintf(fid, 'WORK_DIR="\\$HOME/GeneralizablePyrolysisModel/bpDNN_withAsh4Ea/HyperparametersOptimization"\n');
    fprintf(fid, 'cd \\$WORK_DIR\n');
    fprintf(fid, '\n');
    fprintf(fid, '# Load required modules\n');
    fprintf(fid, 'module load matlab/r2022a\n');
    fprintf(fid, '\n');    fprintf(fid, '# Get configuration ID from SLURM_ARRAY_TASK_ID environment variable\n');
    fprintf(fid, 'CONFIG_ID=\\${SLURM_ARRAY_TASK_ID}\n');
    fprintf(fid, '\n');
    fprintf(fid, '# Ensure we''re using SLURM array job\n');
    fprintf(fid, 'if [ -z "\\$CONFIG_ID" ]; then\n');
    fprintf(fid, '    echo "ERROR: No configuration ID provided. Should be submitted as SLURM array job."\n');
    fprintf(fid, '    exit 1\n');
    fprintf(fid, 'fi\n');
    fprintf(fid, '\n');
    fprintf(fid, '# Output start information\n');
    fprintf(fid, 'echo "Starting hyperparameter optimization configuration ID: \\$CONFIG_ID"\n');
    fprintf(fid, 'echo "Hostname: \\$(hostname)"\n');
    fprintf(fid, 'echo "Running time: \\$(date)"\n');
    fprintf(fid, 'echo "MATLAB version: \\$(matlab -nosplash -nodesktop -r ''version, exit'' | grep MATLAB)"\n');
    fprintf(fid, '\n');
    fprintf(fid, '# Run hyperparameter optimization using MATLAB\n');
    fprintf(fid, 'matlab -nodisplay -nosplash -r "addpath(''\\$WORK_DIR/scripts''); args = {''\\$CONFIG_ID''}; runOptimization; exit"\n');
    fprintf(fid, '\n');
    fprintf(fid, '# Output completion information\n');
    fprintf(fid, 'echo "Configuration ID \\$CONFIG_ID processing complete"\n');
    fprintf(fid, 'echo "Completion time: \\$(date)"\n');
    fprintf(fid, 'EOL\n');
    fprintf(fid, '    \n');    fprintf(fid, '    # Submit SLURM array job\n');
    fprintf(fid, '    echo "Submitting hyperparameter optimization array job, range: $ARRAY_RANGE"\n');
    fprintf(fid, '    JOB_ID=$(sbatch --array=$ARRAY_RANGE $JOB_SCRIPT | awk ''{print $4}'')\n');
    fprintf(fid, '    \n');
    fprintf(fid, '    echo "Jobs submitted! Job ID: $JOB_ID"\n');
    fprintf(fid, '    echo "After all configurations complete, run the following command to analyze results:"\n');
    fprintf(fid, '    echo "  $0 analyze"\n');
    fprintf(fid, '    \n');
    fprintf(fid, '    # Clean up temporary job script\n');
    fprintf(fid, '    rm $JOB_SCRIPT\n');
    fprintf(fid, '}\n\n');
    
    % Analyze results function
    fprintf(fid, '# Function to analyze results\n');
    fprintf(fid, 'analyze_results() {\n');
    fprintf(fid, '    echo "======================================================================"\n');
    fprintf(fid, '    echo "  Hyperparameter Optimization Results Analysis"\n');
    fprintf(fid, '    echo "  $(date)"\n');
    fprintf(fid, '    echo "======================================================================"\n');
    fprintf(fid, '    \n');
    fprintf(fid, '    # Create analysis job script\n');
    fprintf(fid, '    ANALYSIS_SCRIPT="hyperopt_analysis_$$.sh"\n');    fprintf(fid, '    cat > $ANALYSIS_SCRIPT << EOL\n');
    fprintf(fid, '#!/bin/bash\n');
    fprintf(fid, '#SBATCH --job-name=bp_dnn_analyze\n');
    fprintf(fid, '#SBATCH --nodes=1\n');
    fprintf(fid, '#SBATCH --ntasks-per-node=4\n');
    fprintf(fid, '#SBATCH --time=4:00:00\n');
    fprintf(fid, '#SBATCH --output=%%j.out\n');
    fprintf(fid, '#SBATCH --error=%%j.err\n');
    fprintf(fid, '#SBATCH --account=siqi-ic\n');
    fprintf(fid, '#SBATCH --partition=IllinoisComputes\n');
    fprintf(fid, '\n');
    fprintf(fid, '# Set working directory\n');
    fprintf(fid, 'WORK_DIR="\\$HOME/GeneralizablePyrolysisModel/bpDNN_withAsh4Ea/HyperparametersOptimization"\n');
    fprintf(fid, 'cd \\$WORK_DIR\n');
    fprintf(fid, '\n');
    fprintf(fid, '# Load required modules\n');
    fprintf(fid, 'module load matlab/r2022a\n');
    fprintf(fid, '\n');
    fprintf(fid, '# Output start information\n');
    fprintf(fid, 'echo "Starting hyperparameter optimization results analysis"\n');
    fprintf(fid, 'echo "Hostname: \\$(hostname)"\n');
    fprintf(fid, 'echo "Running time: \\$(date)"\n');
    fprintf(fid, '\n');
    fprintf(fid, '# Use MATLAB to analyze results\n');
    fprintf(fid, 'matlab -nodisplay -nosplash -r "addpath(''\\$WORK_DIR/scripts''); args = {}; runOptimization; exit"\n');
    fprintf(fid, '\n');
    fprintf(fid, '# Create results archive\n');
    fprintf(fid, 'TIMESTAMP=\\$(date +"%Y%%m%%d_%%H%%M%%S")\n');
    fprintf(fid, 'RESULT_ARCHIVE="optimization_results_\\${TIMESTAMP}.tar.gz"\n');
    fprintf(fid, '\n');
    fprintf(fid, 'echo "Creating results archive: \\$RESULT_ARCHIVE"\n');
    fprintf(fid, 'tar -czf \\$RESULT_ARCHIVE results/ best_model/ figures/\n');
    fprintf(fid, '\n');
    fprintf(fid, '# Output completion information\n');
    fprintf(fid, 'echo "Analysis complete"\n');
    fprintf(fid, 'echo "Results archived as: \\$RESULT_ARCHIVE"\n');
    fprintf(fid, 'echo "Completion time: \\$(date)"\n');
    fprintf(fid, 'EOL\n');
    fprintf(fid, '    \n');
    fprintf(fid, '    # Submit analysis job\n');
    fprintf(fid, '    JOB_ID=$(qsub $ANALYSIS_SCRIPT)\n');
    fprintf(fid, '    \n');
    fprintf(fid, '    echo "Analysis job submitted! Job ID: $JOB_ID"\n');
    fprintf(fid, '    \n');
    fprintf(fid, '    # Clean up temporary job script\n');
    fprintf(fid, '    rm $ANALYSIS_SCRIPT\n');
    fprintf(fid, '}\n\n');
    
    % Main script execution
    fprintf(fid, '# Main script execution\n');
    fprintf(fid, 'case "$1" in\n');
    fprintf(fid, '    submit)\n');
    fprintf(fid, '        submit_jobs\n');
    fprintf(fid, '        ;;\n');
    fprintf(fid, '    analyze)\n');
    fprintf(fid, '        analyze_results\n');
    fprintf(fid, '        ;;\n');
    fprintf(fid, '    help|--help|-h)\n');
    fprintf(fid, '        print_usage\n');
    fprintf(fid, '        ;;\n');
    fprintf(fid, '    *)\n');
    fprintf(fid, '        echo "ERROR: Unknown command ''$1''"\n');
    fprintf(fid, '        print_usage\n');
    fprintf(fid, '        exit 1\n');
    fprintf(fid, '        ;;\n');
    fprintf(fid, 'esac\n\n');
    fprintf(fid, 'exit 0\n');
    
    % Close file
    fclose(fid);
    
    % Make script executable
    system('chmod +x hyperParamOptimization.sh');
end

% Helper function to visualize results
function visualize_results(results_dir, figures_dir)
    % Load results
    results_file = fullfile(results_dir, 'all_results_sorted.csv');
    results = readtable(results_file);
    
    % Create figures directory if it doesn't exist
    if ~exist(figures_dir, 'dir')
        mkdir(figures_dir);
    end
    
    % Figure 1: MSE comparison
    figure('visible', 'on');
    bar([results.train_mse(1:10), results.val_mse(1:10), results.test_mse(1:10)]);
    title('MSE Comparison for Top 10 Configurations');
    xlabel('Configuration Rank');
    ylabel('Mean Squared Error');
    legend('Training', 'Validation', 'Testing');    grid on;
    saveas(gcf, fullfile(figures_dir, 'MSE_Comparison.fig'));
    print(gcf, fullfile(figures_dir, 'MSE_Comparison.png'), '-dpng', '-r600');
    print(gcf, fullfile(figures_dir, 'MSE_Comparison.eps'), '-depsc2', '-r600');
    
    % Figure 2: R^2 comparison
    figure('visible', 'on');
    bar([results.train_r2(1:10), results.val_r2(1:10), results.test_r2(1:10)]);
    title('R^2 Comparison for Top 10 Configurations');
    xlabel('Configuration Rank');
    ylabel('R^2 Value');
    legend('Training', 'Validation', 'Testing');    grid on;
    saveas(gcf, fullfile(figures_dir, 'R2_Comparison.fig'));
    print(gcf, fullfile(figures_dir, 'R2_Comparison.png'), '-dpng', '-r600');
    print(gcf, fullfile(figures_dir, 'R2_Comparison.eps'), '-depsc2', '-r600');
    
    % Figure 3: Training time comparison
    figure('visible', 'on');
    bar(results.training_time(1:10));
    title('Training Time for Top 10 Configurations');
    xlabel('Configuration Rank');
    ylabel('Training Time (seconds)');    grid on;
    saveas(gcf, fullfile(figures_dir, 'Training_Time.fig'));
    print(gcf, fullfile(figures_dir, 'Training_Time.png'), '-dpng', '-r600');
    print(gcf, fullfile(figures_dir, 'Training_Time.eps'), '-depsc2', '-r600');
    
    % Figure 4: Learning rate vs. MSE
    [unique_lr, ~, idx] = unique(results.learning_rate);
    avg_mse = zeros(size(unique_lr));
    for i = 1:length(unique_lr)
        avg_mse(i) = mean(results.val_mse(results.learning_rate == unique_lr(i)));
    end
    
    figure('visible', 'on');
    plot(unique_lr, avg_mse, 'o-', 'LineWidth', 2);
    title('Average Validation MSE vs. Learning Rate');
    xlabel('Learning Rate');
    ylabel('Average Validation MSE');    grid on;
    saveas(gcf, fullfile(figures_dir, 'LR_vs_MSE.fig'));
    print(gcf, fullfile(figures_dir, 'LR_vs_MSE.png'), '-dpng', '-r600');
    print(gcf, fullfile(figures_dir, 'LR_vs_MSE.eps'), '-depsc2', '-r600');
    
    % Figure 5: Momentum vs. MSE
    [unique_momentum, ~, idx] = unique(results.momentum);
    avg_mse = zeros(size(unique_momentum));
    for i = 1:length(unique_momentum)
        avg_mse(i) = mean(results.val_mse(results.momentum == unique_momentum(i)));
    end
    
    figure('visible', 'on');
    plot(unique_momentum, avg_mse, 'o-', 'LineWidth', 2);
    title('Average Validation MSE vs. Momentum');
    xlabel('Momentum Coefficient');
    ylabel('Average Validation MSE');    grid on;
    saveas(gcf, fullfile(figures_dir, 'Momentum_vs_MSE.fig'));
    print(gcf, fullfile(figures_dir, 'Momentum_vs_MSE.png'), '-dpng', '-r600');
    print(gcf, fullfile(figures_dir, 'Momentum_vs_MSE.eps'), '-depsc2', '-r600');
    
    % Close all figures
    close all;
end
