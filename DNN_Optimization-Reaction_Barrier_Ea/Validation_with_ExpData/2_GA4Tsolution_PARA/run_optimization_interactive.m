% Script to run the improved genetic algorithm optimization
% This script includes the option to continue iteration for each activation energy

% Clear workspace and command window for clean execution
clear;
clc;

% Add paths to improved functions
addpath(pwd);

% Display welcome message
fprintf('===================================================================\n');
fprintf('Pyrolysis Kinetic Parameter Optimization with Interactive Control\n');
fprintf('===================================================================\n\n');

% Get current directory and parent directory
current_dir = pwd;
parent_dir = fileparts(current_dir);

% Create output directory for results
if ~exist('optimization_diagnostics', 'dir')
    mkdir('optimization_diagnostics');
end

% Load predicted activation energy
fprintf('Loading activation energy predictions...\n');
if exist(fullfile(parent_dir, 'Ea_prediction_results.mat'), 'file')
    load(fullfile(parent_dir, 'Ea_prediction_results.mat'));
    Ea_bpDNN = SampleOut_SewageSludge; % Predicted activation energy in J/mol
    fprintf('Successfully loaded Ea predictions. Mean Ea: %.2f J/mol\n', mean(Ea_bpDNN));
else
    error('Cannot find Ea_prediction_results.mat file.');
end

% Load experimental TG data
tg_data_path = fullfile(parent_dir, '0_ValidationData', 'alphaTG_exp.mat');
if exist(tg_data_path, 'file')
    load(tg_data_path);
    fprintf('Successfully loaded experimental TGA data.\n');
else
    error('Cannot find alphaTG_exp.mat file.');
end

% Heating rate and conversion values
beta = 10/60; % Heating rate in K/s (converted from K/min)
alpha = [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99]; % Conversion values

% Calculate G_alpha matrix and ensure correct orientation
fprintf('Calculating conversion models...\n');
G_alpha = zeros(length(alpha), 130); % Pre-allocate with expected dimensions
for i = 1:length(alpha)
    temp = Conversion(alpha(i)); % Call the original Conversion function
    % Ensure consistent dimensions
    if size(temp, 1) > 1
        G_alpha(i, :) = temp(1, :);
    else
        G_alpha(i, 1:size(temp, 2)) = temp;
    end
end

% Display G_alpha dimensions for verification
[rows, cols] = size(G_alpha);
fprintf('G_alpha matrix dimensions: %d rows × %d columns\n', rows, cols);

% Settings for optimization parameters
nvars = 2; % Number of variables [T, A]

% Original lower bounds reference
lb0 = [293 2.5E18;
       313 5.5E18;
       375 2.3E17;
       473 2.5E16;
       573 2.2E15;
       673 2.1E15;
       773 2.0E15;
       800 2.3E15;
       850 2.0E17;
       900 2.3E17];     % [T, A; ...]

% Start parallel pool if not already running
if isempty(gcp('nocreate'))
    fprintf('Starting parallel pool...\n');
    parpool("Processes");
end

% Initialize result arrays with correct dimensions
num_ea = length(Ea_bpDNN);
num_alpha = length(alpha);
num_models = min(cols, 130); % Limit number of models to process

% Pre-allocate arrays with correct dimensions
T = zeros(num_models, 2, num_ea);      % [T, A] for each model and Ea
fval = zeros(num_models, num_ea);      % Fitness value for each optimization

fprintf('\nReady to start optimization with:\n');
fprintf('- %d activation energies\n', num_ea);
fprintf('- %d alpha values\n', num_alpha);
fprintf('- %d conversion models\n\n', num_models);

% Create log file for overall progress
progressLog = fullfile('optimization_diagnostics', 'optimization_progress.log');
fidProgress = fopen(progressLog, 'w');
fprintf(fidProgress, 'Starting optimization on %s\n\n', datestr(now));
fclose(fidProgress);

% Loop through each activation energy
j = 1;
while j <= num_ea
    fidProgress = fopen(progressLog, 'a');
    fprintf(fidProgress, 'Processing activation energy %d of %d: %.2f J/mol\n', j, num_ea, Ea_bpDNN(j));
    fclose(fidProgress);
    
    % Log file for this optimization run
    logFile = fullfile('optimization_diagnostics', sprintf('optimization_log_Ea_%d.txt', j));
    fid = fopen(logFile, 'w');
    fprintf(fid, 'Optimization for Activation Energy %d: %.2f J/mol\n\n', j, Ea_bpDNN(j));
    fclose(fid);
    
    fprintf('\n===================================================================\n');
    fprintf('Starting optimization for activation energy %d of %d: %.2f J/mol\n', j, num_ea, Ea_bpDNN(j));
    fprintf('===================================================================\n');
    
    % Initialize arrays for parallel computation results
    localT = zeros(num_models, 2);
    localFval = zeros(num_models, 1);
    localLogs = cell(num_models, 1);
    
    % Parallel loop through conversion models
    parfor i = 1:num_models
        currentEa = Ea_bpDNN(j);
        
        % Safely get alpha value
        alpha_idx = min(i, num_alpha);
        currentAlpha = alpha(alpha_idx);
        
        % Get target G_alpha value - use the correct indexing
        target_g_alpha = G_alpha(alpha_idx, i);
        
        % Get adaptive bounds based on conversion level
        [lb, ub] = generateAdaptiveBounds_improved(min(j, size(lb0, 1)), currentAlpha, lb0);
        
        % Run optimization
        [X_best, f_best] = multiStageOptimization_improved(currentEa, beta, target_g_alpha, currentAlpha, lb, ub);
        
        % Store results
        localT(i, :) = X_best;
        localFval(i) = f_best;
        
        % Create log info
        optimInfo = sprintf('Alpha = %.2f, Model = %d, Final Error = %.10e\n  Best T = %.2f K, Best A = %.4e\n\n', 
                           currentAlpha, i, f_best, X_best(1), X_best(2));
        localLogs{i} = optimInfo;
    end
    
    % Store results in main arrays
    T(:, :, j) = localT;
    fval(:, j) = localFval;
    
    % Write optimization results to log file
    fid = fopen(logFile, 'a');
    for i = 1:length(localLogs)
        fprintf(fid, '%s', localLogs{i});
    end
    fclose(fid);
    
    % Log completion of this Ea
    fidProgress = fopen(progressLog, 'a');
    fprintf(fidProgress, 'Completed optimization for activation energy %d\n', j);
    fclose(fidProgress);
    
    fprintf('\nCompleted optimization for activation energy %d of %d.\n', j, num_ea);
    
    % Calculate interim temperature solution for this Ea
    Tsol_interim = zeros(length(alpha), num_models);
    for i = 1:num_models
        Tsol_interim(:, i) = T(i, 1, j) * ones(length(alpha), 1);
    end
    
    % Save interim results
    save(sprintf('Results_kinetics_Ea_%d.mat', j), 'T', 'fval', 'Tsol_interim', 'G_alpha', 'alpha', 'Ea_bpDNN', 'alphaTG_exp');
    fprintf('Interim results saved to Results_kinetics_Ea_%d.mat\n', j);
    
    % Ask if we should continue to the next Ea
    if j < num_ea
        response = input('\nContinue to iterate? (y/n): ', 's');
        if ~strcmpi(response, 'y')
            fidProgress = fopen(progressLog, 'a');
            fprintf(fidProgress, 'User requested to stop after activation energy %d\n', j);
            fclose(fidProgress);
            fprintf('Stopping after activation energy %d as requested.\n', j);
            break;
        end
    end
    
    % Move to next activation energy
    j = j + 1;
end

% Calculate final temperature solution
Tsol = zeros(length(alpha), num_models);
for j = 1:min(j, num_ea) % Only process completed Ea values
    for i = 1:num_models
        Tsol(:, i) = T(i, 1, j) * ones(length(alpha), 1);
    end
end

% Save final results
save('Results_kinetics.mat', 'T', 'fval', 'Tsol', 'G_alpha', 'alpha', 'Ea_bpDNN', 'alphaTG_exp');

% Log completion
fidProgress = fopen(progressLog, 'a');
fprintf(fidProgress, 'Optimization completed. Final results saved to Results_kinetics.mat\n');
fclose(fidProgress);

fprintf('\n===================================================================\n');
fprintf('Optimization completed successfully!\n');
fprintf('Final results saved to Results_kinetics.mat\n');
fprintf('===================================================================\n');
