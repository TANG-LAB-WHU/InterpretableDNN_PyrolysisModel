% Improved Genetic Algorithm for Optimization of Kinetic Parameters
% This script calculates pre-exponential factor (A) and temperature (T)
% based on activation energy (Ea) predicted by the neural network model.

% Get current directory and parent directory for correct file paths
current_dir = pwd;
parent_dir = fileparts(current_dir);

% Create output directory for results
if ~exist('optimization_diagnostics', 'dir')
    mkdir('optimization_diagnostics');
end

% Load predicted activation energy from the previous step
fprintf('Attempting to load Ea prediction results from: %s\n', fullfile(parent_dir, 'Ea_prediction_results.mat'));
if exist(fullfile(parent_dir, 'Ea_prediction_results.mat'), 'file')
    load(fullfile(parent_dir, 'Ea_prediction_results.mat'));
    Ea_bpDNN = SampleOut_SewageSludge; % Predicted activation energy in J/mol
    fprintf('Successfully loaded Ea predictions. Mean Ea: %.2f J/mol\n', mean(Ea_bpDNN));
else
    error('Cannot find Ea_prediction_results.mat file. Please ensure Step 2 completed successfully.');
end

% Load experimental TG data
tg_data_path = fullfile(parent_dir, '0_ValidationData', 'alphaTG_exp.mat');
if exist(tg_data_path, 'file')
    load(tg_data_path);
    fprintf('Successfully loaded experimental TGA data.\n');
else
    error('Cannot find alphaTG_exp.mat file at %s', tg_data_path);
end

% Heating rate and conversion values
beta = 10/60; % Heating rate in K/s (converted from K/min)
alpha = [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99]; % Consistent conversion values

% Calculate G_alpha matrix and ensure correct orientation
G_alpha = zeros(length(alpha), 130); % Pre-allocate with expected dimensions
for i = 1:length(alpha)
    temp = Conversion(alpha(i)); % Call the original Conversion function
    % Ensure consistent dimensions by only taking first row if needed
    G_alpha(i, :) = temp(1, :);
end

% Display G_alpha dimensions for verification
[rows, cols] = size(G_alpha);
fprintf('G_alpha matrix dimensions: %d rows × %d columns\n', rows, cols);

% Settings for optimization parameters
nvars = 2; % Number of variables [T, A]
A = [];
b = [];
Aeq = [];
beq = [];
nonlcon = [];

% Original lower bounds for reference
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
          
% Default upper bounds
ub0 = [973 1e25]; % Higher upper bound for A to allow more flexibility

% Start parallel pool if not already running
if isempty(gcp('nocreate'))
    parpool("Processes");
end

% Initialize result arrays with correct dimensions
num_ea = length(Ea_bpDNN);
num_alpha = length(alpha);
num_models = cols; % Number of conversion models

% Pre-allocate arrays with correct dimensions
T = zeros(num_models, 2, num_ea);      % [T, A] for each model and Ea
fval = zeros(num_models, num_ea);      % Fitness value for each optimization

fprintf('Starting optimization with %d activation energies, %d alpha values, and %d models\n', 
        num_ea, num_alpha, num_models);

% Create log file for overall progress
progressLog = fullfile('optimization_diagnostics', 'optimization_progress.log');
fidProgress = fopen(progressLog, 'w');
fprintf(fidProgress, 'Starting optimization on %s\n\n', datestr(now));

% Loop through each activation energy
for j = 1:num_ea
    % Log file for this optimization run
    logFile = fullfile('optimization_diagnostics', sprintf('optimization_log_Ea_%d.txt', j));
    fid = fopen(logFile, 'w');
    fprintf(fid, 'Optimization for Activation Energy %d: %.2f J/mol\n\n', j, Ea_bpDNN(j));
    fprintf(fidProgress, 'Processing activation energy %d of %d: %.2f J/mol\n', j, num_ea, Ea_bpDNN(j));
    
    % Initialize arrays for parallel computation results
    localT = zeros(num_models, 2);
    localFval = zeros(num_models, 1);
    localLogs = cell(num_models, 1);
    
    % For safety, limit the number of models to process
    models_to_process = min(num_models, 130); % Ensure we don't exceed array bounds
    
    % Parallel loop through conversion models
    parfor i = 1:models_to_process
        currentEa = Ea_bpDNN(j);
        
        % Safely get alpha value (ensure we don't exceed array bounds)
        alpha_idx = min(i, num_alpha);
        currentAlpha = alpha(alpha_idx);
        
        % Get target G_alpha value - use the correct indexing
        target_g_alpha = G_alpha(alpha_idx, i);
        
        % Get adaptive bounds based on conversion level
        [lb, ub] = generateAdaptiveBounds(min(j, size(lb0, 1)), currentAlpha, lb0);
        
        % Run optimization
        [X_best, f_best] = multiStageOptimization(currentEa, beta, target_g_alpha, currentAlpha, lb, ub);
        
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
    for i = 1:length(localLogs)
        fprintf(fid, '%s', localLogs{i});
    end
    fclose(fid);
    
    % Log completion of this Ea
    fprintf(fidProgress, 'Completed optimization for activation energy %d\n', j);
    
    % Ask if we should continue to the next Ea
    if j < num_ea
        fprintf('Completed optimization for activation energy %d of %d. Continue to next? (y/n): ', j, num_ea);
        response = input('', 's');
        if ~strcmpi(response, 'y')
            fprintf(fidProgress, 'User requested to stop after activation energy %d\n', j);
            break;
        end
    end
end

% Calculate temperature solution
Tsol = zeros(length(alpha), models_to_process);
for j = 1:num_ea
    for i = 1:models_to_process
        Tsol(:, i) = T(i, 1, j) * ones(length(alpha), 1);
    end
end

% Save results
save('Results_kinetics.mat', 'T', 'fval', 'Tsol', 'G_alpha', 'alpha', 'Ea_bpDNN', 'alphaTG_exp');
fprintf(fidProgress, 'Optimization completed. Results saved to Results_kinetics.mat\n');
fclose(fidProgress);

fprintf('Optimization completed successfully. Results saved to Results_kinetics.mat\n');
