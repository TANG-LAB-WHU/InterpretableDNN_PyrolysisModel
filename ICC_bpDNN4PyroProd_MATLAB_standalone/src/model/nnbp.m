function [net, gradient, dW, db] = nnbp(net, input, target, error, outlayer, pdW, pdb)
    % NNBP with Gradient Clipping
    
    % --- Input Validation (Basic) ---
    if ~isstruct(net) || ~isfield(net, 'numLayer') || ~isfield(net, 'layer') || ~isfield(net, 'weight') || ~isfield(net, 'bias')
        error('NNBP:InvalidInput', 'Input `net` must be a valid network structure.');
    end
    if ~iscell(outlayer) || length(outlayer) ~= net.numLayer
        error('NNBP:InvalidInput', 'Input `outlayer` must be a cell array with length equal to net.numLayer.');
    end
    % Add more checks if necessary (e.g., sizes of input, target, error)
    % --- End Input Validation ---
    
    % --- Gradient Clipping Threshold ---
    % Adjust this value based on experimentation. Start with 1.0 or 5.0.
    gradient_clip_threshold = 5.0;
    clipping_applied_flag = false; % Flag to track if clipping occurred
    % ---
    
    Q = size(input, 2); % Number of samples in the batch
    if Q == 0
        warning('NNBP:ZeroSamples', 'Input contains zero samples (Q=0). Skipping backpropagation.');
        gradient = NaN; dW = pdW; db = pdb; % Return previous deltas or empty
        return;
    end
    
    Nl = net.numLayer;
    d = cell(1, Nl); % Preallocate cell array for derivatives
    
    % --- Calculate delta (d) for the Output Layer (Layer Nl) ---
    output_activation = outlayer{Nl};
    output_transferFcn = net.layer{Nl}.transferFcn;
    switch lower(output_transferFcn)
        case {'logsig'} % Sigmoid derivative: f'(x) = f(x) * (1 - f(x))
            grad_output = output_activation .* (1 - output_activation);
            d{Nl} = grad_output .* error;
            gradient_signal = grad_output;
        case {'tansig'} % Tanh derivative: f'(x) = 1 - f(x)^2
            grad_output = 1 - output_activation.^2;
            d{Nl} = grad_output .* error;
            gradient_signal = grad_output;
        case {'purelin', 'softmax'} % Linear derivative: f'(x) = 1
            d{Nl} = error;
            if Nl > 1
                last_hidden_activation = outlayer{Nl-1};
                last_hidden_transferFcn = net.layer{Nl-1}.transferFcn;
                switch lower(last_hidden_transferFcn)
                     case {'logsig'}, gradient_signal = last_hidden_activation .* (1 - last_hidden_activation);
                     case {'tansig'}, gradient_signal = 1 - last_hidden_activation.^2;
                     case {'poslin'}, gradient_signal = double(last_hidden_activation > 0);
                     otherwise, warning('NNBP:UnknownGradFcn', 'Unknown transfer function %s for gradient calc. Using 1.', last_hidden_transferFcn); gradient_signal = ones(size(last_hidden_activation));
                end
            else
                 gradient_signal = ones(size(output_activation));
            end
        case {'poslin'} % ReLU derivative: f'(x) = 1 if x > 0, 0 otherwise
            grad_output = double(output_activation > 0);
            d{Nl} = grad_output .* error;
            gradient_signal = grad_output;
        otherwise
            warning('NNBP:UnknownTransferFcn', 'Unknown transfer function "%s" on output layer. Using linear derivative.', output_transferFcn);
            d{Nl} = error;
            if Nl > 1, last_hidden_activation = outlayer{Nl-1}; gradient_signal = double(last_hidden_activation > 0);
            else, gradient_signal = ones(size(output_activation)); end
    end
    
    % --- Backpropagate delta (d) through Hidden Layers ---
    for i = (Nl - 1) : -1 : 1
        hidden_activation = outlayer{i};
        hidden_transferFcn = net.layer{i}.transferFcn;
        switch lower(hidden_transferFcn)
            case {'logsig'}, grad_hidden = hidden_activation .* (1 - hidden_activation);
            case {'tansig'}, grad_hidden = 1 - hidden_activation.^2;
            case {'poslin'}, grad_hidden = double(hidden_activation > 0);
            otherwise, warning('NNBP:UnknownTransferFcn', 'Unknown transfer function "%s" on hidden layer %d. Using linear derivative.', hidden_transferFcn, i); grad_hidden = ones(size(hidden_activation));
        end
        W_next = net.weight{i + 1}'; d_next = d{i + 1};
        if size(W_next, 2) ~= size(d_next, 1), error('NNBP:DimensionMismatch', 'Dimension mismatch calculating delta for layer %d.', i); end
        d{i} = grad_hidden .* (W_next * d_next);
    end
    
    % --- Compute Weight and Bias Deltas (dW, db) ---
    dW = cell(1, Nl); db = cell(1, Nl);
    X_prevLayer = input;
    for i = 1 : Nl
        d_current = d{i};
        if size(d_current, 2) ~= size(X_prevLayer, 2), error('NNBP:SampleCountMismatch', 'Sample count mismatch calculating dW/db for layer %d.', i); end
        if size(d_current, 2) ~= Q, error('NNBP:CriticalError', 'Sample count Q (%d) does not match d_current columns (%d) at layer %d.', Q, size(d_current, 2), i); end
        dW{i} = (d_current * X_prevLayer') / Q;
        db{i} = sum(d_current, 2) / Q;
        if i < Nl, X_prevLayer = outlayer{i}; end
    end
    
    % % --- Dynamic Learning Rate Adjustment ---
    % persistent prev_perf_nnbp;
    % if isempty(prev_perf_nnbp)
    %     current_perf_nnbp = sum(sum(error.^2))/numel(error);
    %     if isnan(current_perf_nnbp) || isinf(current_perf_nnbp), current_perf_nnbp = realmax; end % Handle initial NaN
    %     prev_perf_nnbp = current_perf_nnbp;
    % else
    %     current_perf_nnbp = sum(sum(error.^2))/numel(error);
    %      if isnan(current_perf_nnbp) || isinf(current_perf_nnbp)
    %         % If current perf is NaN/Inf, drastically reduce LR
    %         fprintf('Warning: NaN/Inf performance detected in LR adjustment. Reducing LR.\n');
    %         current_perf_nnbp = prev_perf_nnbp * 10; % Treat as much worse performance
    %      end
    
    %     lr_inc = net.trainParam.lr_inc; lr_dec = net.trainParam.lr_dec;
    %     current_lr = net.trainParam.lr; new_lr = current_lr;
    %     MIN_LR = 1e-8; MAX_LR = 10;
    
    %     if current_perf_nnbp < (prev_perf_nnbp * 0.999) && isfield(net.trainParam, 'lr_inc')
    %         new_lr = current_lr * lr_inc;
    %     elseif current_perf_nnbp > (prev_perf_nnbp * 1.04) && isfield(net.trainParam, 'lr_dec')
    %         new_lr = current_lr / lr_dec;
    %     end
    
    %     new_lr = max(MIN_LR, min(MAX_LR, new_lr));
    %     net.trainParam.lr = new_lr;
    %     prev_perf_nnbp = current_perf_nnbp; % Store possibly large finite value if NaN occurred
    % end
    % % --- End Dynamic Learning Rate ---

    % --- Dynamic Learning Rate Adjustment ---
    persistent prev_perf_nnbp;
    if isempty(prev_perf_nnbp)
        current_perf_nnbp = sum(sum(error.^2))/numel(error);
        if isnan(current_perf_nnbp) || isinf(current_perf_nnbp), current_perf_nnbp = realmax; end
        prev_perf_nnbp = current_perf_nnbp;
    else
        current_perf_nnbp = sum(sum(error.^2))/numel(error);
        lr_inc = net.trainParam.lr_inc;
        lr_dec = net.trainParam.lr_dec;
        current_lr = net.trainParam.lr;
        new_lr = current_lr; % Start with current LR

        MIN_LR = 1e-8;
        MAX_LR = 1.05; % <<<< LOWER MAX LR CAP <<<<

        % Check for NaN/Inf or large explosion first
        if isnan(current_perf_nnbp) || isinf(current_perf_nnbp) || current_perf_nnbp > (prev_perf_nnbp * 2.0) % Check if performance doubled (or NaN/Inf)
            fprintf('Warning: NaN/Inf or large performance increase detected. Drastically reducing LR.\n');
            new_lr = max(MIN_LR, current_lr / 5.0); % Divide LR by 5, ensuring it doesn't go below MIN_LR
            if isnan(current_perf_nnbp) || isinf(current_perf_nnbp), current_perf_nnbp = prev_perf_nnbp * 10; end % Assign large value for prev_perf storage
        else
            % Apply normal dynamic adjustment if stable
            if current_perf_nnbp < (prev_perf_nnbp * 0.999) && isfield(net.trainParam, 'lr_inc')
                new_lr = current_lr * lr_inc;
            elseif current_perf_nnbp > (prev_perf_nnbp * 1.04) && isfield(net.trainParam, 'lr_dec')
                new_lr = current_lr / lr_dec; % Original decrease logic
            end
        end

        % Apply bounds and update network LR
        new_lr = max(MIN_LR, min(MAX_LR, new_lr)); % Clamp LR within [MIN_LR, MAX_LR]
        % Only print if LR actually changed (and is not NaN)
        if ~isnan(new_lr) && new_lr ~= current_lr 
            % fprintf('DEBUG nnbp: Dynamic LR adjusted from %.4e to %.4e\n', current_lr, new_lr);
        end
        if ~isnan(new_lr), net.trainParam.lr = new_lr; end % Update LR only if it's a valid number

        % Store current performance for next iteration's comparison
        prev_perf_nnbp = current_perf_nnbp;
    end
    % --- End Dynamic Learning Rate ---


    % --- Apply Momentum, Gradient Clipping, Learning Rate and Update ---
    lr = net.trainParam.lr;
    mc = net.trainParam.mc;
    
    for i = 1 : Nl
        % Apply Regularization
        rw = 1.0; rb = 1.0;
        if isfield(net, 'layer') && isfield(net.layer{i}, 'rw') && ~isempty(net.layer{i}.rw), rw = net.layer{i}.rw; end
        if isfield(net, 'layer') && isfield(net.layer{i}, 'rb') && ~isempty(net.layer{i}.rb), rb = net.layer{i}.rb; end
        dW{i} = rw .* dW{i};
        db{i} = rb .* db{i};
    
        % Apply Momentum
        if mc > 0
            if ~iscell(pdW) || length(pdW) < i || ~isequal(size(dW{i}), size(pdW{i})), pdW{i} = zeros(size(dW{i})); end
            if ~iscell(pdb) || length(pdb) < i || ~isequal(size(db{i}), size(pdb{i})), pdb{i} = zeros(size(db{i})); end
            dW{i} = mc * pdW{i} + (1 - mc) * dW{i};
            db{i} = mc * pdb{i} + (1 - mc) * db{i};
        end
    
        % --- Gradient Clipping (Applied BEFORE LR multiplication) ---
        % Calculate the L2 norm of the combined gradients for weights and biases
        grad_norm_w = norm(dW{i}(:), 2);
        grad_norm_b = norm(db{i}(:), 2);
        % Using separate norms might be more stable than combined norm
        % total_grad_norm = sqrt(grad_norm_w^2 + grad_norm_b^2); % Optional combined norm
    
        if grad_norm_w > gradient_clip_threshold
            clipping_applied_flag = true;
            scale_factor = gradient_clip_threshold / grad_norm_w;
            dW{i} = dW{i} * scale_factor;
            % fprintf('DEBUG NNBP: Clipping dW{%d} norm %.2f -> %.2f\n', i, grad_norm_w, norm(dW{i}(:), 2)); % Optional debug
        end
         if grad_norm_b > gradient_clip_threshold
            clipping_applied_flag = true;
            scale_factor = gradient_clip_threshold / grad_norm_b;
            db{i} = db{i} * scale_factor;
            % fprintf('DEBUG NNBP: Clipping db{%d} norm %.2f -> %.2f\n', i, grad_norm_b, norm(db{i}(:), 2)); % Optional debug
         end
        % --- End Gradient Clipping ---
    
        % Weight/Bias Update
        if ~isfield(net, 'weight') || length(net.weight) < i || ~isequal(size(net.weight{i}), size(dW{i})), error('NNBP:UpdateError', 'Weight matrix size mismatch for layer %d during update.', i); end
        if ~isfield(net, 'bias') || length(net.bias) < i || ~isequal(size(net.bias{i}), size(db{i})), error('NNBP:UpdateError', 'Bias vector size mismatch for layer %d during update.', i); end
    
        % Check for NaN/Inf in deltas BEFORE update
        if any(isnan(dW{i}(:))) || any(isinf(dW{i}(:))) || any(isnan(db{i}(:))) || any(isinf(db{i}(:)))
            warning('NNBP:NaNInfDelta', 'NaN or Inf detected in dW or db for layer %d BEFORE update. Skipping update for this layer.', i);
            % Keep pdW/pdb as they were before this iteration's calculation if skipping update
            dW{i} = pdW{i}; % Revert dW to previous delta
            db{i} = pdb{i}; % Revert db to previous delta
            continue; % Skip the update for this layer
        end
    
        net.weight{i} = net.weight{i} - lr * dW{i}; % Standard gradient descent
        net.bias{i} = net.bias{i} - lr * db{i};
    end
    
    if clipping_applied_flag
        % fprintf('DEBUG NNBP: Gradient clipping was applied this epoch.\n'); % Optional debug
    end
    
    % --- Final Gradient Calculation for Stop Criteria ---
    gradient = mean(abs(gradient_signal(:)));
    if isnan(gradient) || isinf(gradient)
        warning('NNBP:GradientNaNInf', 'Calculated gradient for stopping criteria is NaN or Inf.');
        gradient = realmax('double'); % Return large value if NaN/Inf
    end
    
    end % End nnbp function

% % function [net, gradient, dW, db] = nnbp(net, input, target, error, outlayer, pdW, pdb)
% % % NNBP progressively calibrates weights and biases of the net according the
% % %       BP learning algorithm. Return gradient of output layer if output
% % %       transferFcn is logsig. Otherwise return gradient of last hidden
% % %       layer if output transferFcn is purelin.
% % %   Inputs:
% % %           net -  A feedforward neural network.
% % %           input - Input of networks.
% % %           error - Errors between targets and outputs of network.
% % %           pdW - Previous delta of weights (W).
% % %           pdb - Previous delta of biases (b).
% % %   output:
% % %           net - The updated network
% % %           gradient - gradient of the network. When transfer function on
% % %   output layer is purelin, the average gradient of nerons of last hidden
% % %   layer is returned. Otherwise, the average gradient of neurons of output
% % %   layer is returned.
% % %           dW - Delta weights for this iteration.
% % %           db - Delta biases for this iteration.
% % %   Example:
% % %           [net, gradient, dW, db] = NNBP(net, input, target, error, outlayer, pdW, pdb)
% % % the number of batch samples (batch size).
% % Q = size(input, 2);

% % Nl =  net.numLayer;
% % switch net.layer{Nl}.transferFcn
% %     case {'logsig'}
% %         gradient = outlayer{Nl} .* (1 - outlayer{Nl});
% %         d{Nl} =  gradient .* error;
% %     case {'purelin', 'softmax'}
% %         gradient = outlayer{Nl - 1} .* (1 - outlayer{Nl - 1});
% %         d{Nl} = error;
% %     case {'poslin'} % ReLU derivative: 1 for x>0, 0 for x<=0
% %         % Create a binary mask where outputs > 0
% %         mask = outlayer{Nl} > 0;
% %         d{Nl} = error .* mask;
% % end

% % for i = Nl - 1 : -1 : 1
% %     switch net.layer{i}.transferFcn
% %         case {'logsig'}
% %             grad = outlayer{i} .* (1 - outlayer{i});
% %         case {'tansig'}
% %             grad = 1 - outlayer{i} .^ 2;
% %         case {'tansigopt'}
% %             grad = 1.7159 * 2/3 * (1 - 1/(1.7159)^2 * outlayer{i} .^ 2);
% %         case {'poslin'} % ReLU derivative: 1 for x>0, 0 for x<=0
% %             % Create a binary mask where outputs > 0
% %             mask = outlayer{i} > 0;
% %             grad = mask; % For ReLU, gradient is 1 where x>0, 0 elsewhere

% %             % --- START INSERTION 4 (Inside nnbp.m, first loop, only needed if using poslin) ---
% %             if strcmp(net.layer{i}.transferFcn, 'poslin') % Check only necessary for poslin case in this specific code structure
% %                 W_term = net.weight{i + 1}';
% %                 d_term = d{i + 1};
% %                 fprintf('DEBUG nnbp (d{%d} calc): Size W'' = [%d, %d], Size d{%d} = [%d, %d]\n', ...
% %                         i, size(W_term,1), size(W_term,2), i+1, size(d_term,1), size(d_term,2));
% %                 if size(W_term, 2) ~= size(d_term, 1)
% %                     fprintf('ERROR nnbp (d{%d} calc): Dimension mismatch detected BEFORE multiplication!\n', i);
% %                     % Handle error appropriately
% %                 end
% %             end
% %             % --- END INSERTION 4 ---

% %             d{i} = grad .* (net.weight{i + 1}' * d{i + 1});
% %     end
% % end

% % % Compute dW and db.
% % X = input;
% % for i = Nl : -1 : 1
% %     if i == 1

% %         % --- START INSERTION 5 (Inside nnbp.m, second loop, i==1 case) ---
% %         d_term = d{i};
% %         X_term = X'; % Note: X is 'input' here
% %         fprintf('DEBUG nnbp (dW{%d} calc): Size d{%d} = [%d, %d], Size X'' = [%d, %d]\n', ...
% %                 i, i, size(d_term,1), size(d_term,2), size(X_term,1), size(X_term,2));
% %         if size(d_term, 2) ~= size(X_term, 1)
% %              fprintf('ERROR nnbp (dW{%d} calc): Dimension mismatch detected BEFORE multiplication (i=1)!\n', i);
% %              % Handle error appropriately
% %         end
% %         % --- END INSERTION 5 ---

% %         dW{i} = (d{i} * X') / Q;
% %     else
% %         % --- START INSERTION 6 (Inside nnbp.m, second loop, i>1 case) ---
% %         d_term = d{i};
% %         out_term = outlayer{i - 1}';
% %         fprintf('DEBUG nnbp (dW{%d} calc): Size d{%d} = [%d, %d], Size outlayer{%d}'' = [%d, %d]\n', ...
% %                 i, i, size(d_term,1), size(d_term,2), i-1, size(out_term,1), size(out_term,2));
% %         if size(d_term, 2) ~= size(out_term, 1)
% %             fprintf('ERROR nnbp (dW{%d} calc): Dimension mismatch detected BEFORE multiplication (i>1)!\n', i);
% %             % Handle error appropriately
% %         end
% %         % --- END INSERTION 6 ---
% %         dW{i} = (d{i} * outlayer{i - 1}') / Q;
% %     end
% %         db{i} = d{i} * ones(1, Q)' / Q;
% % end

% % % Checek gradients
% % % net = nncheckgrad(net, input, target, dW, db); % removal for speed up

% % % Dynamic learning rate adjustment
% % % Check if we have previous values to compare
% % persistent prev_perf;
% % if isempty(prev_perf)
% %     prev_perf = sum(sum(error.^2))/numel(error); % MSE
% % else
% %     current_perf = sum(sum(error.^2))/numel(error);
    
% %     % If performance improves, increase learning rate
% %     if current_perf < prev_perf && isfield(net.trainParam, 'lr_inc')
% %         net.trainParam.lr = net.trainParam.lr * net.trainParam.lr_inc;
% %     % If performance deteriorates, decrease learning rate
% %     elseif current_perf > prev_perf * 1.04 && isfield(net.trainParam, 'lr_dec') % 4% tolerance
% %         net.trainParam.lr = net.trainParam.lr / net.trainParam.lr_dec;
% %     end
    
% %     % Store current performance for next iteration
% %     prev_perf = current_perf;
% % end

% % % Apply learning rate
% % lr = net.trainParam.lr;

% % % update weights and biases.
% % for i = Nl : -1 : 1
    
% %     rw = net.layer{i}.rw;
% %     rb = net.layer{i}.rb;
% %     dW{i} = rw .* dW{i};
% %     db{i} = rb .* db{i};
    
% %     mc = net.trainParam.mc;
% %     if mc > 0
% %         dW{i} = mc * pdW{i} + (1 - mc) * dW{i};
% %         db{i} = mc * pdb{i} + (1 - mc) * db{i};
% %     end
    
% %     % Apply learning rate when updating weights and biases
% %     net.weight{i} = net.weight{i} + lr * dW{i};
% %     net.bias{i} = net.bias{i} + lr * db{i};
% % end

% % % Average gradient.
% % gradient = sum(sum(gradient)) / (size(gradient, 1) * size(gradient, 2));

% function [net, gradient, dW, db] = nnbp(net, input, target, error, outlayer, pdW, pdb)
%     % NNBP progressively calibrates weights and biases of the net according the
%     %       BP learning algorithm. Return gradient of output layer if output
%     %       transferFcn is logsig. Otherwise return gradient of last hidden
%     %       layer if output transferFcn is purelin.
%     %   Inputs:
%     %           net -  A feedforward neural network.
%     %           input - Input of networks (Features x Samples).
%     %           target - Target outputs (Outputs x Samples).
%     %           error - Errors between targets and outputs of network (Outputs x Samples).
%     %           outlayer - Cell array of activations for ALL layers from nnff (LayerActivations x Samples).
%     %           pdW - Previous delta of weights (W) (Cell array).
%     %           pdb - Previous delta of biases (b) (Cell array).
%     %   output:
%     %           net - The updated network
%     %           gradient - Average gradient magnitude used for stopping criteria.
%     %           dW - Delta weights for this iteration (Cell array).
%     %           db - Delta biases for this iteration (Cell array).
%     %   Example:
%     %           [net, gradient, dW, db] = NNBP(net, input, target, error, outlayer, pdW, pdb)
    
%     % --- Input Validation (Basic) ---
%     if ~isstruct(net) || ~isfield(net, 'numLayer') || ~isfield(net, 'layer') || ~isfield(net, 'weight') || ~isfield(net, 'bias')
%         error('NNBP:InvalidInput', 'Input `net` must be a valid network structure.');
%     end
%     if ~iscell(outlayer) || length(outlayer) ~= net.numLayer
%         error('NNBP:InvalidInput', 'Input `outlayer` must be a cell array with length equal to net.numLayer.');
%     end
%     % Add more checks if necessary (e.g., sizes of input, target, error)
%     % --- End Input Validation ---
    
    
%     Q = size(input, 2); % Number of samples in the batch
%     if Q == 0
%         warning('NNBP:ZeroSamples', 'Input contains zero samples (Q=0). Skipping backpropagation.');
%         gradient = NaN; dW = pdW; db = pdb; % Return previous deltas or empty
%         return;
%     end
    
%     Nl = net.numLayer;
%     d = cell(1, Nl); % Preallocate cell array for derivatives
    
%     % --- Calculate delta (d) for the Output Layer (Layer Nl) ---
%     output_activation = outlayer{Nl};
%     output_transferFcn = net.layer{Nl}.transferFcn;
%     switch lower(output_transferFcn)
%         case {'logsig'} % Sigmoid derivative: f'(x) = f(x) * (1 - f(x))
%             grad_output = output_activation .* (1 - output_activation);
%             d{Nl} = grad_output .* error;
%             % Gradient for stop criteria uses output layer gradient for logsig
%             gradient_signal = grad_output;
%         case {'tansig'} % Tanh derivative: f'(x) = 1 - f(x)^2
%             grad_output = 1 - output_activation.^2;
%             d{Nl} = grad_output .* error;
%              % Gradient for stop criteria uses output layer gradient for tansig
%             gradient_signal = grad_output;
%         case {'purelin', 'softmax'} % Linear derivative: f'(x) = 1
%             % For purelin/softmax output, delta is just the error signal
%             d{Nl} = error;
%             % Gradient for stop criteria uses LAST HIDDEN layer gradient
%             if Nl > 1
%                 gradient_signal = outlayer{Nl - 1} .* (1 - outlayer{Nl - 1}); % Assumes last hidden is logsig - NEEDS FIX
%                 % --- FIX: Calculate gradient based on ACTUAL last hidden layer ---
%                 last_hidden_activation = outlayer{Nl-1};
%                 last_hidden_transferFcn = net.layer{Nl-1}.transferFcn;
%                 switch lower(last_hidden_transferFcn)
%                      case {'logsig'}, gradient_signal = last_hidden_activation .* (1 - last_hidden_activation);
%                      case {'tansig'}, gradient_signal = 1 - last_hidden_activation.^2;
%                      case {'poslin'}, gradient_signal = double(last_hidden_activation > 0); % Use double() for logical->numeric
%                      otherwise, warning('NNBP:UnknownGradFcn', 'Unknown transfer function %s for gradient calc. Using 1.', last_hidden_transferFcn); gradient_signal = ones(size(last_hidden_activation));
%                 end
%                 % --- END FIX ---
%             else % Network has no hidden layers
%                  gradient_signal = ones(size(output_activation)); % Treat as linear gradient
%             end
%         case {'poslin'} % ReLU derivative: f'(x) = 1 if x > 0, 0 otherwise
%             grad_output = double(output_activation > 0); % Use double() for logical->numeric
%             d{Nl} = grad_output .* error;
%             % Gradient for stop criteria uses output layer gradient for poslin
%             gradient_signal = grad_output;
%         otherwise
%             warning('NNBP:UnknownTransferFcn', 'Unknown transfer function "%s" on output layer. Using linear derivative.', output_transferFcn);
%             d{Nl} = error; % Default to linear behavior
%             if Nl > 1 % Try to use last hidden layer gradient
%                  last_hidden_activation = outlayer{Nl-1};
%                  gradient_signal = double(last_hidden_activation > 0); % Default guess: poslin
%             else
%                  gradient_signal = ones(size(output_activation)); % Default guess: linear
%             end
%     end
    
%     % --- Backpropagate delta (d) through Hidden Layers ---
%     for i = (Nl - 1) : -1 : 1
%         hidden_activation = outlayer{i};
%         hidden_transferFcn = net.layer{i}.transferFcn;
    
%         % Calculate gradient of the hidden layer's activation function
%         switch lower(hidden_transferFcn)
%             case {'logsig'}
%                 grad_hidden = hidden_activation .* (1 - hidden_activation);
%             case {'tansig'}
%                 grad_hidden = 1 - hidden_activation.^2;
%             % case {'tansigopt'} % This seems non-standard, ensure definition is correct if used
%             %     grad_hidden = 1.7159 * 2/3 * (1 - 1/(1.7159)^2 * hidden_activation.^2);
%             case {'poslin'} % ReLU derivative
%                 grad_hidden = double(hidden_activation > 0); % Use double() for logical->numeric
%             otherwise
%                  warning('NNBP:UnknownTransferFcn', 'Unknown transfer function "%s" on hidden layer %d. Using linear derivative.', hidden_transferFcn, i);
%                  grad_hidden = ones(size(hidden_activation)); % Default to linear behavior
%         end
    
%         % Calculate delta for the current hidden layer
%         W_next = net.weight{i + 1}'; % Transpose weights from NEXT layer
%         d_next = d{i + 1};       % Delta from NEXT layer
    
%         % --- Dimension Check before Multiplication ---
%         if size(W_next, 2) ~= size(d_next, 1)
%             fprintf('ERROR nnbp (d{%d} calc): Dimension mismatch! Size W_next'' = [%d, %d], Size d_next = [%d, %d]\n', ...
%                     i, size(W_next,1), size(W_next,2), size(d_next,1), size(d_next,2));
%             error('NNBP:DimensionMismatch', 'Dimension mismatch calculating delta for layer %d.', i);
%         end
%         % --- End Dimension Check ---
    
%         d{i} = grad_hidden .* (W_next * d_next);
%     end
    
%     % --- Compute Weight and Bias Deltas (dW, db) ---
%     dW = cell(1, Nl); % Preallocate
%     db = cell(1, Nl); % Preallocate
%     X_prevLayer = input; % Input to the first layer
    
%     for i = 1 : Nl % Iterate forward through layers to calculate dW/db
%         d_current = d{i}; % Delta for the current layer
    
%         % --- Dimension Check before Multiplication ---
%          if size(d_current, 2) ~= size(X_prevLayer, 2)
%              fprintf('ERROR nnbp (dW/db{%d} calc): Sample count mismatch! Size d_current = [%d, %d], Size X_prevLayer = [%d, %d]\n', ...
%                      i, size(d_current,1), size(d_current,2), size(X_prevLayer,1), size(X_prevLayer,2));
%              error('NNBP:SampleCountMismatch', 'Sample count mismatch calculating dW/db for layer %d.', i);
%          end
%          if size(d_current, 2) ~= Q % Double check against initial Q
%              error('NNBP:CriticalError', 'Sample count Q (%d) does not match d_current columns (%d) at layer %d.', Q, size(d_current, 2), i);
%          end
%         % --- End Dimension Check ---
    
%         % Calculate dW and db for the current layer
%         % dW{i} = (delta{i} * activation{i-1}') / numSamples; (General form)
%         % db{i} = sum(delta{i}, 2) / numSamples; (General form)
%         dW{i} = (d_current * X_prevLayer') / Q;
%         db{i} = sum(d_current, 2) / Q; % Sum deltas across samples, then average
    
%         % Update X_prevLayer for the next iteration (activation of current layer)
%         if i < Nl
%             X_prevLayer = outlayer{i};
%         end
%     end
    
    
%     % --- Dynamic Learning Rate Adjustment ---
%     % NOTE: Using persistent variables can be tricky. Consider passing state if needed.
%     persistent prev_perf_nnbp; % Use a name specific to this function scope
%     if isempty(prev_perf_nnbp)
%         % Initialize with current MSE if first call within a session/clear
%         current_perf_nnbp = sum(sum(error.^2))/numel(error); % Use error matrix passed in
%         prev_perf_nnbp = current_perf_nnbp; % Initialize previous with current
%     else
%         current_perf_nnbp = sum(sum(error.^2))/numel(error); % Calculate current MSE
    
%         lr_inc = net.trainParam.lr_inc;
%         lr_dec = net.trainParam.lr_dec;
%         current_lr = net.trainParam.lr;
%         new_lr = current_lr; % Start with current LR
    
%         % Define bounds for learning rate to prevent extreme values
%         MIN_LR = 1e-8;
%         MAX_LR = 10; % Adjust MAX_LR based on your problem
    
%         % If performance improves significantly, increase learning rate
%         if current_perf_nnbp < (prev_perf_nnbp * 0.999) && isfield(net.trainParam, 'lr_inc') % Require some improvement
%             new_lr = current_lr * lr_inc;
%         % If performance deteriorates significantly, decrease learning rate
%         elseif current_perf_nnbp > (prev_perf_nnbp * 1.04) && isfield(net.trainParam, 'lr_dec') % 4% tolerance for worsening
%             new_lr = current_lr / lr_dec; % Use division for decrease factor
%         end
    
%         % Apply bounds and update network LR
%         new_lr = max(MIN_LR, min(MAX_LR, new_lr)); % Clamp LR
%         if new_lr ~= current_lr
%              % fprintf('DEBUG nnbp: LR adjusted from %.4e to %.4e (Perf: %.4e -> %.4e)\n', current_lr, new_lr, prev_perf_nnbp, current_perf_nnbp); % Optional debug
%         end
%         net.trainParam.lr = new_lr;
    
%         % Store current performance for next iteration's comparison
%         prev_perf_nnbp = current_perf_nnbp;
%     end
%     % --- End Dynamic Learning Rate ---
    
    
%     % --- Apply Momentum and Learning Rate to Update Weights and Biases ---
%     lr = net.trainParam.lr; % Get potentially adjusted LR
%     mc = net.trainParam.mc; % Get momentum coefficient
    
%     for i = 1 : Nl % Iterate forward through layers for update
    
%         % --- Apply Regularization Weight/Bias Factors (if defined) ---
%         % These seem specific to your implementation (rw, rb)
%         rw = 1.0; rb = 1.0; % Default values
%         if isfield(net, 'layer') && isfield(net.layer{i}, 'rw') && ~isempty(net.layer{i}.rw), rw = net.layer{i}.rw; end
%         if isfield(net, 'layer') && isfield(net.layer{i}, 'rb') && ~isempty(net.layer{i}.rb), rb = net.layer{i}.rb; end
%         dW{i} = rw .* dW{i};
%         db{i} = rb .* db{i};
%         % --- End Regularization ---
    
%         % --- Apply Momentum ---
%         if mc > 0
%             % Check if previous deltas (pdW, pdb) have the correct size
%             if ~iscell(pdW) || length(pdW) < i || ~isequal(size(dW{i}), size(pdW{i}))
%                 % Initialize previous delta if needed (e.g., first iteration)
%                 % fprintf('DEBUG nnbp: Initializing pdW{%d} for momentum.\n', i); % Optional debug
%                 pdW{i} = zeros(size(dW{i}));
%             end
%             if ~iscell(pdb) || length(pdb) < i || ~isequal(size(db{i}), size(pdb{i}))
%                 % fprintf('DEBUG nnbp: Initializing pdb{%d} for momentum.\n', i); % Optional debug
%                 pdb{i} = zeros(size(db{i}));
%             end
%             % Apply momentum update rule
%             dW{i} = mc * pdW{i} + (1 - mc) * dW{i};
%             db{i} = mc * pdb{i} + (1 - mc) * db{i};
%         end
%         % --- End Momentum ---
    
    
%         % --- Apply Learning Rate and Update Network ---
%         % Ensure weights/biases exist and sizes match deltas
%         if ~isfield(net, 'weight') || length(net.weight) < i || ~isequal(size(net.weight{i}), size(dW{i}))
%              error('NNBP:UpdateError', 'Weight matrix size mismatch for layer %d during update.', i);
%         end
%          if ~isfield(net, 'bias') || length(net.bias) < i || ~isequal(size(net.bias{i}), size(db{i}))
%              error('NNBP:UpdateError', 'Bias vector size mismatch for layer %d during update.', i);
%         end
    
%         % Update weights and biases
%         net.weight{i} = net.weight{i} - lr * dW{i}; % ** Standard update is SUBTRACTION **
%         net.bias{i} = net.bias{i} - lr * db{i};   % ** Standard update is SUBTRACTION **
%         % --- End Update ---
%     end
    
%     % --- Calculate final gradient magnitude for stopping criteria ---
%     % Use the 'gradient_signal' calculated earlier based on output transfer function
%     gradient = mean(abs(gradient_signal(:))); % Mean absolute value of relevant gradient signals
    
%     % Check for NaN/Inf in final gradient
%     if isnan(gradient) || isinf(gradient)
%         warning('NNBP:GradientNaNInf', 'Calculated gradient for stopping criteria is NaN or Inf.');
%         % Optionally set to a large value to prevent premature stopping if needed
%         % gradient = realmax('double');
%     end
    
%     end
% 