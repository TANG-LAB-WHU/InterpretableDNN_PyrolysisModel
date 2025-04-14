function  [outLayer, error, loss] = nnff(net, input, target)
% NNFF retrieves output of layers according to weights and
%                   biases in forward propagation phrase and error (loss).
%   net = NNFEEDFORWARD(net, input, target) Feed inputs (in) to
%                           neural network (net), and calculate
%                           outputs of separate layers by turn.

% Handle case where target is not provided
if nargin < 3
    target = [];
end

% Check for empty input or target
if isempty(input) || size(input, 2) == 0
    % Return empty structures with NaN values to prevent dimension errors
    outLayer = cell(net.numLayer, 1);
    for k = 1:net.numLayer
        outLayer{k} = [];
    end
    error = [];
    loss = NaN;
    return;
end

% the number of batch samples (batch size).
Q = size(input, 2); 

% number of layers (numHiddenLayers + numOutputs).
Nl = net.numLayer;

% the output of input layer of the network is same as the input of input layer.
out = input;

% Initialize outLayer cell array
outLayer = cell(Nl, 1);

% calculate output of hidden layer of the network.
for i = 1 : Nl - 1
    Wb = [net.weight{i}, net.bias{i}];
    X = [out; ones(1, Q)];
    
    % Check matrix dimensions before multiplication
    [wb_rows, wb_cols] = size(Wb);
    [x_rows, x_cols] = size(X);
    if wb_cols ~= x_rows
        % Dimension mismatch - return NaN results instead of error
        outLayer = cell(net.numLayer, 1);
        for k = 1:net.numLayer
            outLayer{k} = [];
        end
        error = [];
        loss = NaN;
        return;
    end
    
    switch net.layer{i}.transferFcn
        case {'logsig'}
            out = logsig(Wb*X);
        case {'tansig'}
            out = tansig(Wb*X);
        case {'tansigopt'} % not available
            out = tansigopt(Wb*X);
        case {'poslin'} % ReLU (Rectified Linear Unit)
            out = max(0, Wb*X);
    end
    outLayer{i} = out;
end

% calculate output of output layer of the network.
Wb = [net.weight{Nl}, net.bias{Nl}];
X = [outLayer{Nl -1}; ones(1, Q)];

% Check matrix dimensions before final layer multiplication
[wb_rows, wb_cols] = size(Wb);
[x_rows, x_cols] = size(X);
if wb_cols ~= x_rows
    % Dimension mismatch - return NaN results instead of error
    outLayer = cell(net.numLayer, 1);
    for k = 1:net.numLayer
        outLayer{k} = [];
    end
    error = [];
    loss = NaN;
    return;
end

switch net.layer{Nl}.transferFcn
    case {'logsig'}
        outLayer{Nl} = logsig(Wb*X);
    case {'purelin'}
        outLayer{Nl} = purelin(Wb*X);
    case {'softmax'}
        outLayer{Nl} = softmax(Wb*X);
    case {'poslin'} % ReLU (Rectified Linear Unit)
        outLayer{Nl} = max(0, Wb*X);
end

% Only calculate error and loss if target is provided
if ~isempty(target) && size(target, 2) == size(input, 2)
    output = outLayer{Nl};
    error = target - output;

    % Initialize loss to NaN in case the loss calculation code below fails
    loss = NaN;

    try
        switch net.layer{Nl}.transferFcn
            case {'logsig', 'purelin'}
                loss = 1/2 * sum(sum(error .^ 2)) / Q;
            case {'softmax'}
                loss = -sum(sum(target .* log(output))) / Q;
        end
    catch
        % If loss calculation fails, just keep the NaN value
        warning('NeuralNet:LossCalculationFailed', 'Failed to calculate loss value');
    end
else
    % If no target, return empty error and NaN loss
    error = [];
    loss = NaN;
end