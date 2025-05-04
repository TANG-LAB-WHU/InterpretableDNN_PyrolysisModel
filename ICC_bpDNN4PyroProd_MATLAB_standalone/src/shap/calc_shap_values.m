function [shap_values, feature_names, X, target_names] = calc_shap_values(model, X_data, target_names, feature_names, options)
% CALC_SHAP_VALUES Compute SHAP values for neural network model
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
        
        % Check for layer-based network structure
        if isstruct(model) && isfield(model, 'layer')
            preds = forward_pass_layer(model, X);
            pred = preds(:, target_idx);
            return;
        end
        
        % Last resort: try a direct call assuming model is a function handle
        if isa(model, 'function_handle')
            preds = model(X);
            pred = preds(:, target_idx);
            return;
        end
        
        % If we get here, we couldn't identify the model type
        error('Unrecognized model type. Please implement a custom prediction function.');
        
    catch ME
        % In case of error, provide helpful diagnostic information
        error('Error in predict_target: %s.\nModel fields: %s', ME.message, ...
              strjoin(fieldnames(model), ', '));
    end
end

% Helper function for W/b style networks
function preds = forward_pass_wb(model, X)
    % Perform forward pass for network with W/b structure
    num_samples = size(X, 1);
    a = X; % Initial activation is the input
    
    for layer_idx = 1:length(model.W)
        % For each sample, apply the layer transformation
        z = zeros(num_samples, size(model.b{layer_idx}, 1));
        
        for i = 1:num_samples
            z(i, :) = (model.W{layer_idx} * a(i, :)')' + model.b{layer_idx}';
        end
        
        % Apply activation function
        if layer_idx < length(model.W) % Hidden layer
            if isfield(model, 'transferFcn') && length(model.transferFcn) >= layer_idx
                a = apply_activation(z, model.transferFcn{layer_idx});
            else
                a = tanh(z); % Default to tanh for hidden layers
            end
        else % Output layer
            if isfield(model, 'transferFcn') && length(model.transferFcn) >= layer_idx
                a = apply_activation(z, model.transferFcn{layer_idx});
            else
                a = z; % Default to linear for output layer
            end
        end
    end
    
    preds = a;
end

% Helper function for weight/bias style networks
function preds = forward_pass_weight_bias(model, X)
    % Perform forward pass for network with weight/bias structure
    num_samples = size(X, 1);
    a = X; % Initial activation is the input
    
    if iscell(model.weight)
        num_layers = length(model.weight);
    else
        num_layers = size(model.weight, 1);
    end
    
    for layer_idx = 1:num_layers
        % Get weights and biases for this layer
        if iscell(model.weight)
            W = model.weight{layer_idx};
            b = model.bias{layer_idx};
        else
            W = model.weight(layer_idx);
            b = model.bias(layer_idx);
        end
        
        % Forward propagation
        z = zeros(num_samples, size(b, 1));
        for i = 1:num_samples
            z(i, :) = (W * a(i, :)')' + b';
        end
        
        % Apply activation
        if layer_idx < num_layers % Hidden layer
            if isfield(model, 'transferFcn') && length(model.transferFcn) >= layer_idx
                a = apply_activation(z, model.transferFcn{layer_idx});
            else
                a = tanh(z); % Default to tanh for hidden layers
            end
        else % Output layer
            if isfield(model, 'transferFcn') && length(model.transferFcn) >= layer_idx
                a = apply_activation(z, model.transferFcn{layer_idx});
            else
                a = z; % Default to linear for output layer
            end
        end
    end
    
    preds = a;
end

% Helper function for layer-based networks
function preds = forward_pass_layer(model, X)
    % Perform forward pass for network with layer structure
    num_samples = size(X, 1);
    a = X; % Initial activation is the input
    
    for layer_idx = 1:length(model.layer)
        layer = model.layer{layer_idx};
        
        % Get weights and biases
        if isfield(layer, 'weight') && isfield(layer, 'bias')
            W = layer.weight;
            b = layer.bias;
            
            % Forward propagation
            z = zeros(num_samples, size(b, 1));
            for i = 1:num_samples
                z(i, :) = (W * a(i, :)')' + b';
            end
            
            % Apply activation
            if isfield(layer, 'transferFcn')
                a = apply_activation(z, layer.transferFcn);
            else
                if layer_idx < length(model.layer)
                    a = tanh(z); % Default for hidden layers
                else
                    a = z; % Default for output layer
                end
            end
        else
            error('Layer %d missing weight/bias fields', layer_idx);
        end
    end
    
    preds = a;
end

% Helper function to apply activation functions
function a = apply_activation(z, fcn_name)
    switch lower(fcn_name)
        case 'tansig'
            a = tanh(z);
        case 'logsig'
            a = 1./(1 + exp(-z));
        case 'poslin' % ReLU
            a = max(0, z);
        case 'purelin'
            a = z;
        case 'softmax'
            exp_z = exp(z - max(z, [], 2)); % Subtract max for numerical stability
            a = exp_z ./ sum(exp_z, 2);
        case 'elliotsig'
            a = z ./ (1 + abs(z));
        case 'satlin'
            a = min(1, max(0, z));
        case 'radbas'
            a = exp(-z.^2);
        otherwise
            warning('Unknown activation function "%s". Using linear activation.', fcn_name);
            a = z;
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
    current_diff = sample_prediction - expected_value;
    shap_sum = sum(shap_values);
    
    if abs(shap_sum) > 1e-10  % Avoid division by zero
        shap_values = shap_values * (current_diff / shap_sum);
    end
end

function coalitions = generate_coalitions(num_features, num_samples)
    % Generate a set of coalition samples for SHAP calculation
    % This follows recommended sampling strategies from the official SHAP methodology
    
    % Always include empty coalition and full coalition
    empty_coalition = zeros(1, num_features);
    full_coalition = ones(1, num_features);
    
    % Generate random coalitions for the remaining samples
    if num_samples > 2
        % For large feature spaces, we want a more structured sampling approach
        if num_features > 10 && num_samples > 100
            % Allocate coalitions array
            random_coalitions = zeros(num_samples-2, num_features);
            
            % First, include some small and large coalitions (recommended by SHAP authors)
            % as these are more informative for Shapley values
            num_structured = min(num_samples-2, num_features*2);
            
            for i = 1:num_structured/2
                % Small coalitions (1-3 features active)
                small_size = randi(3);
                small_coal = zeros(1, num_features);
                small_indices = randperm(num_features, small_size);
                small_coal(small_indices) = 1;
                
                % Large coalitions (n-3 to n-1 features active)
                large_size = num_features - randi(3);
                large_coal = zeros(1, num_features);
                large_indices = randperm(num_features, large_size);
                large_coal(large_indices) = 1;
                
                % Store these structured coalitions
                if i*2-1 <= size(random_coalitions, 1)
                    random_coalitions(i*2-1, :) = small_coal;
                end
                if i*2 <= size(random_coalitions, 1)
                    random_coalitions(i*2, :) = large_coal;
                end
            end
            
            % Fill remainder with uniform random coalitions
            if num_structured < num_samples-2
                random_coalitions((num_structured+1):end, :) = rand(num_samples-2-num_structured, num_features) > 0.5;
            end
        else
            % For smaller feature spaces, uniform random sampling is adequate
            random_coalitions = rand(num_samples-2, num_features) > 0.5;
        end
        
        coalitions = [empty_coalition; random_coalitions; full_coalition];
    else
        coalitions = [empty_coalition; full_coalition];
    end
end

function weights = calculate_kernel_weights(coalitions, num_features)
    % Calculate kernel SHAP weights for each coalition
    
    % Get coalition sizes (number of 1's in each coalition)
    coalition_sizes = sum(coalitions, 2);
    
    % Calculate weights based on Shapley kernel formula
    weights = zeros(size(coalitions, 1), 1);
    
    for i = 1:length(weights)
        % Skip empty and full coalitions to avoid division by zero
        if coalition_sizes(i) == 0 || coalition_sizes(i) == num_features
            weights(i) = 1;
        else
            % Original Shapley kernel formula:
            % w_i = (M-1) / (binom(M,|z'|) * |z'| * (M-|z'|))
            % where |z'| is the number of non-zero elements in coalition
            weights(i) = (num_features - 1) / ...
                (nchoosek(num_features, coalition_sizes(i)) * ...
                coalition_sizes(i) * (num_features - coalition_sizes(i)));
        end
    end
end

function mixed_samples = generate_mixed_sample(x, background, coalition)
    % Generate mixed samples for a coalition
    % Features in coalition come from x, others from background
    
    num_bg = size(background, 1);
    mixed_samples = zeros(num_bg, length(coalition));
    
    % For each feature:
    % If feature is in coalition (1), use the value from x
    % If feature is not in coalition (0), use values from background
    for j = 1:length(coalition)
        if coalition(j)
            mixed_samples(:, j) = repmat(x(j), num_bg, 1);
        else
            mixed_samples(:, j) = background(:, j);
        end
    end
end