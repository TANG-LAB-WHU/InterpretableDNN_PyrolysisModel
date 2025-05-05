function [shap_values, feature_names, X, target_names] = shap_kernel_function(model, X_data, target_names, feature_names, options)
% SHAP_KERNEL_FUNCTION Compute SHAP values for neural network model
%
% INPUTS:
%   model - Neural network model (contains weights, biases, etc.)
%   X_data - Features data matrix [num_samples x num_features]
%   target_names - Cell array of target variable names
%   feature_names - Cell array of feature names
%   options - Structure with calculation parameters:
%       .num_background - Number of background samples (default: 100)
%       .num_samples - Number of samples for SHAP calculation (default: all)
%       .background_data - Optional background dataset (if not provided, X_data is used)
%       .num_permutations - Number of permutations for Shapley approx (default: 100)
%       .parallel - Use parallel processing (default: true)
%       .verbose - Show progress information (default: true)
%
% OUTPUTS:
%   shap_values - Cell array of SHAP values for each target [num_samples x num_features]
%   feature_names - Feature names used in the calculation
%   X - Data samples used for SHAP calculation
%   target_names - Target names used in the calculation
%
% METHODOLOGY:
%   This function implements the Kernel SHAP method following the approach in 
%   "A Unified Approach to Interpreting Model Predictions" by Lundberg & Lee (2017)
%   with optimizations for neural networks.
%
% Version 2.0.0 - Enhanced implementation with proper SHAP coalitional sampling

    % Start timer
    tic;
    
    % Default parameters
    if nargin < 5
        options = struct();
    end
    
    % Get data dimensions
    [num_samples, num_features] = size(X_data);
    
    % Set default options if not provided
    if ~isfield(options, 'num_background')
        options.num_background = min(100, num_samples);
    end
    
    if ~isfield(options, 'num_samples')
        options.num_samples = num_samples;
    else
        options.num_samples = min(options.num_samples, num_samples);
    end
    
    if ~isfield(options, 'background_data')
        % If no background data provided, use X_data
        options.background_data = X_data;
    end
    
    if ~isfield(options, 'num_permutations')
        options.num_permutations = 100; % Default permutations for approximation
    end
    
    if ~isfield(options, 'parallel')
        options.parallel = true;
    end
    
    if ~isfield(options, 'verbose')
        options.verbose = true;
    end
    
    % Check if feature names are provided
    if nargin < 4 || isempty(feature_names)
        feature_names = cell(1, num_features);
        for i = 1:num_features
            feature_names{i} = sprintf('Feature %d', i);
        end
    end
    
    % Check target names
    if nargin < 3 || isempty(target_names)
        target_names = {'Target 1'};
    end
    
    % Sample data for SHAP calculation if needed
    if options.num_samples < num_samples
        sample_indices = randsample(num_samples, options.num_samples);
        X = X_data(sample_indices, :);
    else
        X = X_data;
    end
    
    % Sample background data
    if size(options.background_data, 1) > options.num_background
        bg_indices = randsample(size(options.background_data, 1), options.num_background);
        background = options.background_data(bg_indices, :);
    else
        background = options.background_data;
    end
    
    if options.verbose
        fprintf('Computing SHAP values: %d samples, %d features, %d background samples\n', ...
            options.num_samples, num_features, size(background, 1));
    end
    
    % Generate predictions using the model
    % First for the samples
    try
        X_predictions = nnpredict(model, X);
    catch ME
        error('Error predicting with model: %s', ME.message);
    end
    
    % Then for the background
    try
        background_predictions = nnpredict(model, background);
    catch ME
        error('Error predicting with model for background samples: %s', ME.message);
    end
    
    % Calculate expected value (average prediction on background data)
    expected_values = mean(background_predictions, 1);
    
    % Get number of outputs/targets
    num_targets = size(X_predictions, 2);
    
    % Update target_names if needed
    if length(target_names) < num_targets
        for i = (length(target_names)+1):num_targets
            target_names{i} = sprintf('Target %d', i);
        end
    elseif length(target_names) > num_targets
        target_names = target_names(1:num_targets);
    end
    
    % Initialize cell array for SHAP values (one cell per target)
    shap_values = cell(1, num_targets);
    
    % Start parallel pool if using parallel processing
    if options.parallel
        if options.verbose
            fprintf('Setting up parallel pool...\n');
        end
        try
            pool = gcp('nocreate');
            if isempty(pool)
                parpool('local');
            end
        catch ME
            warning('%s', ['Error creating parallel pool: ' ME.message ' Continuing without parallel processing.']);
            options.parallel = false;
        end
    end
    
    % Main loop for calculating SHAP values for each target
    for target_idx = 1:num_targets
        if options.verbose
            fprintf('Calculating SHAP values for target %d of %d: %s\n', ...
                target_idx, num_targets, target_names{target_idx});
        end
        
        % Initialize SHAP values for this target
        target_shap = zeros(size(X, 1), num_features);
        
        % Create anonymous function to get model prediction for specific target
        % This simplifies the code as we're working with one target at a time
        predict_fn = @(x) predict_target(model, x, target_idx);
        
        % Calculate reference value (expected value for this target)
        ref_value = expected_values(target_idx);
        
        % Loop through each sample to compute SHAP values
        if options.parallel
            % Parallel version
            parfor i = 1:size(X, 1)
                x_sample = X(i, :);
                target_shap(i, :) = kernel_shap_sample(predict_fn, x_sample, background, ...
                    ref_value, options.num_permutations);
                
                % Progress reporting inside parfor is limited, so we don't show it
            end
        else
            % Serial version with progress reporting
            for i = 1:size(X, 1)
                x_sample = X(i, :);
                target_shap(i, :) = kernel_shap_sample(predict_fn, x_sample, background, ...
                    ref_value, options.num_permutations);
                
                % Show progress every 10 samples
                if options.verbose && mod(i, 10) == 0
                    fprintf('  Processed %d of %d samples (%.1f%%)\n', ...
                        i, size(X, 1), 100 * i / size(X, 1));
                end
            end
        end
        
        % Store the SHAP values for this target
        shap_values{target_idx} = target_shap;
        
        if options.verbose
            fprintf('Completed SHAP calculations for %s\n', target_names{target_idx});
        end
    end
    
    % Report calculation time
    if options.verbose
        fprintf('SHAP calculation completed in %.2f seconds\n', toc);
    end
end

function pred = predict_target(model, X, target_idx)
    % Predict for a specific target with robust handling of different model types
    
    % Check if X needs transposing (expected format is samples × features)
    if size(X, 2) == 1 && size(X, 1) > 1
        % X is likely a column vector when it should be a row vector
        X = X';
    end
    
    % Try different prediction methods based on model structure
    try
        % First try standard nnpredict if available
        if exist('nnpredict', 'file') == 2
            preds = nnpredict(model, X);
            pred = preds(:, target_idx);
            return;
        end
        
        % Check if it's a MATLAB neural network object
        if isobject(model) && ismethod(model, 'sim')
            preds = sim(model, X')'; % MATLAB expects features × samples, then transpose result
            pred = preds(:, target_idx);
            return;
        end
        
        % Check for custom network structure with W and b fields
        if isstruct(model) && isfield(model, 'W') && isfield(model, 'b')
            % Perform forward pass
            preds = forward_pass_wb(model, X);
            pred = preds(:, target_idx);
            return;
        end
        
        % Check for network with weight and bias fields
        if isstruct(model) && isfield(model, 'weight') && isfield(model, 'bias')
            preds = forward_pass_weight_bias(model, X);
            pred = preds(:, target_idx);
            return;
        end
        
        % If none of the above methods work, throw an error
        error('Unsupported model structure for prediction.');
    catch ME
        error('Error during prediction: %s', ME.message);
    end
end

function shap_values = kernel_shap_sample(predict_fn, x, background, expected_value, num_permutations)
    % Implementation of Kernel SHAP for a single sample
    
    % Get dimensionality
    num_features = length(x);
    
    % Generate coalitions (sets of features)
    % For computational efficiency, we sample coalitions rather than using all 2^M possibilities
    coalitions = generate_coalitions(num_features, num_permutations);
    
    % Initialize arrays for Shapley regression
    X_shap = zeros(size(coalitions, 1), num_features + 1);
    X_shap(:, 1) = 1; % First column is for the intercept
    
    % Calculate weights for each coalition
    coalition_weights = calculate_kernel_weights(coalitions, num_features);
    
    % Evaluate model for each coalition
    y = zeros(size(coalitions, 1), 1);
    
    for i = 1:size(coalitions, 1)
        coalition = coalitions(i, :);
        
        % Create mixed sample for this coalition: 
        % features in coalition come from x, others from background
        z = generate_mixed_sample(x, background, coalition);
        
        % Predict with the provided function
        y(i) = mean(predict_fn(z));
        
        % Store coalition in SHAP regression design matrix
        X_shap(i, 2:end) = coalition;
    end
    
    % Solve the weighted least squares problem: min_φ ||W^(1/2)(Xφ - y)||_2^2
    % where W is the diagonal matrix of coalition_weights
    
    % Apply weights to both X and y
    weighted_X = X_shap .* sqrt(coalition_weights);
    weighted_y = y .* sqrt(coalition_weights);
    
    % Solve using normal equations for weighted least squares
    phi = (weighted_X' * weighted_X) \ (weighted_X' * weighted_y);
    
    % Extract SHAP values (skip the intercept term)
    shap_values = phi(2:end)';
    
    % Adjust to ensure sum of SHAP values equals difference from expected value
    sample_prediction = predict_fn(x);
    prediction_diff = sample_prediction - expected_value;
    sum_shap = sum(shap_values);
    
    % Scale SHAP values to match the prediction difference
    if abs(sum_shap) > 1e-10
        shap_values = shap_values * (prediction_diff / sum_shap);
    end
end

function coalitions = generate_coalitions(num_features, num_permutations)
    % Generate coalition sampling for SHAP calculation
    
    % Always include empty coalition and full coalition
    coalitions = [zeros(1, num_features); ones(1, num_features)];
    
    % Add randomly sampled coalitions for the approximation
    if num_features <= 10
        % For small feature counts, we can just use all possible coalitions
        all_coalitions = de2bi((0:2^num_features-1)', num_features);
        coalitions = all_coalitions;
    else
        % For larger feature counts, we sample coalitions
        remaining_samples = num_permutations - 2;
        
        % Generate random coalition sizes (number of 1s in each coalition)
        % We want a distribution that emphasizes smaller and larger coalitions
        % to better capture interactions
        coalition_sizes = randi(num_features, remaining_samples, 1);
        
        % Generate the actual coalitions
        sampled_coalitions = zeros(remaining_samples, num_features);
        for i = 1:remaining_samples
            % Select coalition_sizes(i) random features to include
            features_to_include = randperm(num_features, coalition_sizes(i));
            sampled_coalitions(i, features_to_include) = 1;
        end
        
        % Combine with empty and full coalitions
        coalitions = [coalitions; sampled_coalitions];
    end
end

function weights = calculate_kernel_weights(coalitions, num_features)
    % Calculate kernel weights for SHAP value calculation
    % These weights are based on coalition sizes to properly implement Shapley values
    
    % Get the number of active features in each coalition
    coalition_sizes = sum(coalitions, 2);
    
    % Apply the kernel formula: (M-1) / (binom(M, k) * k * (M-k))
    % where M is the number of features, k is the coalition size
    weights = zeros(size(coalition_sizes));
    
    for i = 1:length(coalition_sizes)
        k = coalition_sizes(i);
        
        % Handle edge cases to avoid division by zero
        if k == 0 || k == num_features
            weights(i) = 1;
        else
            % Calculate binomial coefficient
            binom_coef = nchoosek(num_features, k);
            
            % Apply the formula
            weights(i) = (num_features - 1) / (binom_coef * k * (num_features - k));
        end
    end
end

function mixed_samples = generate_mixed_sample(x, background, coalition)
    % Generate mixed samples for coalition
    % For features in coalition, use values from x
    % For features not in coalition, use values from background
    
    num_background = size(background, 1);
    
    % Create a matrix where each row is a mixed sample
    mixed_samples = zeros(num_background, length(x));
    
    for i = 1:num_background
        % Start with background values
        mixed_sample = background(i, :);
        
        % Replace values for features in the coalition
        mixed_sample(coalition == 1) = x(coalition == 1);
        
        % Add to the result
        mixed_samples(i, :) = mixed_sample;
    end
end

function preds = forward_pass_wb(model, X)
    % Perform forward pass for a neural network with W and b fields
    
    % Get network structure
    if isfield(model, 'numLayers')
        num_layers = model.numLayers;
    else
        num_layers = length(model.W);
    end
    
    % Initial activation is the input
    a = X;
    
    % Forward pass through all layers
    for i = 1:num_layers
        if isfield(model.W, ['layer' num2str(i)])
            % Handle struct with layerN fields
            w = model.W.(['layer' num2str(i)]);
            b = model.b.(['layer' num2str(i)]);
        else
            % Handle cell array or direct indexing
            if iscell(model.W)
                w = model.W{i};
                b = model.b{i};
            else
                w = model.W(i);
                b = model.b(i);
            end
        end
        
        % Calculate pre-activation
        z = a * w + b;
        
        % Apply activation function (assume sigmoid for hidden layers, linear for output)
        if i < num_layers
            a = sigmoid(z);
        else
            a = z; % Linear activation for output layer
        end
    end
    
    % Return final layer activations as predictions
    preds = a;
end

function preds = forward_pass_weight_bias(model, X)
    % Perform forward pass for a neural network with weight and bias fields
    
    % Determine the number of layers
    if isfield(model, 'numLayers')
        num_layers = model.numLayers;
    else
        % Try to infer from structure
        if isfield(model, 'weights')
            if iscell(model.weights)
                num_layers = length(model.weights);
            else
                error('Cannot determine number of layers from model structure');
            end
        else
            error('Cannot determine number of layers from model structure');
        end
    end
    
    % Initial activation is the input
    a = X;
    
    % Forward pass through all layers
    for i = 1:num_layers
        if iscell(model.weights)
            w = model.weights{i};
            b = model.biases{i};
        else
            % Try to access through indexing or fields
            try
                w = model.weights(i);
                b = model.biases(i);
            catch
                error('Unable to access weights and biases from model structure');
            end
        end
        
        % Calculate pre-activation
        z = a * w + b;
        
        % Apply activation function (assume sigmoid for hidden layers, linear for output)
        if i < num_layers
            a = sigmoid(z);
        else
            a = z; % Linear activation for output layer
        end
    end
    
    % Return final layer activations as predictions
    preds = a;
end

function y = sigmoid(x)
    % Sigmoid activation function
    y = 1 ./ (1 + exp(-x));
end 