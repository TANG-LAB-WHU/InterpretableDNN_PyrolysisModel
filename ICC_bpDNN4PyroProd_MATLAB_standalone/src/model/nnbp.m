function [net, gradient, dW, db] = nnbp(net, input, target, error, outlayer, pdW, pdb)
% NNBP progressively calibrates weights and biases of the net according the
%       BP learning algorithm. Return gradient of output layer if output
%       transferFcn is logsig. Otherwise return gradient of last hidden
%       layer if output transferFcn is purelin.
%   Inputs:
%           net -  A feedforward neural network.
%           input - Input of networks.
%           error - Errors between targets and outputs of network.
%           pdW - Previous delta of weights (W).
%           pdb - Previous delta of biases (b).
%   output:
%           net - The updated network
%           gradient - gradient of the network. When transfer function on
%   output layer is purelin, the average gradient of nerons of last hidden
%   layer is returned. Otherwise, the average gradient of neurons of output
%   layer is returned.
%           dW - Delta weights for this iteration.
%           db - Delta biases for this iteration.
%   Example:
%           [net, gradient, dW, db] = NNBP(net, input, target, error, outlayer, pdW, pdb)
% the number of batch samples (batch size).
Q = size(input, 2);

Nl =  net.numLayer;
switch net.layer{Nl}.transferFcn
    case {'logsig'}
        gradient = outlayer{Nl} .* (1 - outlayer{Nl});
        d{Nl} =  gradient .* error;
    case {'purelin', 'softmax'}
        gradient = outlayer{Nl - 1} .* (1 - outlayer{Nl - 1});
        d{Nl} = error;
    case {'poslin'} % ReLU derivative: 1 for x>0, 0 for x<=0
        % Create a binary mask where outputs > 0
        mask = outlayer{Nl} > 0;
        d{Nl} = error .* mask;
end

for i = Nl - 1 : -1 : 1
    switch net.layer{i}.transferFcn
        case {'logsig'}
            grad = outlayer{i} .* (1 - outlayer{i});
        case {'tansig'}
            grad = 1 - outlayer{i} .^ 2;
        case {'tansigopt'}
            grad = 1.7159 * 2/3 * (1 - 1/(1.7159)^2 * outlayer{i} .^ 2);
        case {'poslin'} % ReLU derivative: 1 for x>0, 0 for x<=0
            % Create a binary mask where outputs > 0
            mask = outlayer{i} > 0;
            grad = mask; % For ReLU, gradient is 1 where x>0, 0 elsewhere

            % --- START INSERTION 4 (Inside nnbp.m, first loop, only needed if using poslin) ---
            if strcmp(net.layer{i}.transferFcn, 'poslin') % Check only necessary for poslin case in this specific code structure
                W_term = net.weight{i + 1}';
                d_term = d{i + 1};
                fprintf('DEBUG nnbp (d{%d} calc): Size W'' = [%d, %d], Size d{%d} = [%d, %d]\n', ...
                        i, size(W_term,1), size(W_term,2), i+1, size(d_term,1), size(d_term,2));
                if size(W_term, 2) ~= size(d_term, 1)
                    fprintf('ERROR nnbp (d{%d} calc): Dimension mismatch detected BEFORE multiplication!\n', i);
                    % Handle error appropriately
                end
            end
            % --- END INSERTION 4 ---

            d{i} = grad .* (net.weight{i + 1}' * d{i + 1});
    end
end

% Compute dW and db.
X = input;
for i = Nl : -1 : 1
    if i == 1

        % --- START INSERTION 5 (Inside nnbp.m, second loop, i==1 case) ---
        d_term = d{i};
        X_term = X'; % Note: X is 'input' here
        fprintf('DEBUG nnbp (dW{%d} calc): Size d{%d} = [%d, %d], Size X'' = [%d, %d]\n', ...
                i, i, size(d_term,1), size(d_term,2), size(X_term,1), size(X_term,2));
        if size(d_term, 2) ~= size(X_term, 1)
             fprintf('ERROR nnbp (dW{%d} calc): Dimension mismatch detected BEFORE multiplication (i=1)!\n', i);
             % Handle error appropriately
        end
        % --- END INSERTION 5 ---

        dW{i} = (d{i} * X') / Q;
    else
        % --- START INSERTION 6 (Inside nnbp.m, second loop, i>1 case) ---
        d_term = d{i};
        out_term = outlayer{i - 1}';
        fprintf('DEBUG nnbp (dW{%d} calc): Size d{%d} = [%d, %d], Size outlayer{%d}'' = [%d, %d]\n', ...
                i, i, size(d_term,1), size(d_term,2), i-1, size(out_term,1), size(out_term,2));
        if size(d_term, 2) ~= size(out_term, 1)
            fprintf('ERROR nnbp (dW{%d} calc): Dimension mismatch detected BEFORE multiplication (i>1)!\n', i);
            % Handle error appropriately
        end
        % --- END INSERTION 6 ---
        dW{i} = (d{i} * outlayer{i - 1}') / Q;
    end
        db{i} = d{i} * ones(1, Q)' / Q;
end

% Checek gradients
% net = nncheckgrad(net, input, target, dW, db); % removal for speed up

% Dynamic learning rate adjustment
% Check if we have previous values to compare
persistent prev_perf;
if isempty(prev_perf)
    prev_perf = sum(sum(error.^2))/numel(error); % MSE
else
    current_perf = sum(sum(error.^2))/numel(error);
    
    % If performance improves, increase learning rate
    if current_perf < prev_perf && isfield(net.trainParam, 'lr_inc')
        net.trainParam.lr = net.trainParam.lr * net.trainParam.lr_inc;
    % If performance deteriorates, decrease learning rate
    elseif current_perf > prev_perf * 1.04 && isfield(net.trainParam, 'lr_dec') % 4% tolerance
        net.trainParam.lr = net.trainParam.lr / net.trainParam.lr_dec;
    end
    
    % Store current performance for next iteration
    prev_perf = current_perf;
end

% Apply learning rate
lr = net.trainParam.lr;

% update weights and biases.
for i = Nl : -1 : 1
    
    rw = net.layer{i}.rw;
    rb = net.layer{i}.rb;
    dW{i} = rw .* dW{i};
    db{i} = rb .* db{i};
    
    mc = net.trainParam.mc;
    if mc > 0
        dW{i} = mc * pdW{i} + (1 - mc) * dW{i};
        db{i} = mc * pdb{i} + (1 - mc) * db{i};
    end
    
    % Apply learning rate when updating weights and biases
    net.weight{i} = net.weight{i} + lr * dW{i};
    net.bias{i} = net.bias{i} + lr * db{i};
end

% Average gradient.
gradient = sum(sum(gradient)) / (size(gradient, 1) * size(gradient, 2));