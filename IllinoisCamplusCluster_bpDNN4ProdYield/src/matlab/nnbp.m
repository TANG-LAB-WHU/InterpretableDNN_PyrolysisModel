function [net, gradient, perf] = nnbp(net, input, target)
% NNBP Implements the backpropagation algorithm for neural networks
% Inputs:
%   net - Neural network structure
%   input - Input data
%   target - Target data
% Outputs:
%   net - Updated neural network
%   gradient - Network gradient
%   perf - Performance metric (MSE)

% Forward pass to get network output
[layerOutputs, networkOutput] = nnff(net, input);
output = networkOutput;

% Calculate error and performance
error = target - output;
perf = mean(mean(error.^2)); % MSE

% Get batch size
batchSize = size(input, 2);

% Initialize gradient storage
numLayers = length(net.layer);
d = cell(numLayers, 1);

% Calculate output layer gradient
switch net.layer{numLayers}.transferFcn
    case 'logsig'
        outputGrad = output .* (1 - output);
        d{numLayers} = outputGrad .* error;
        gradient = outputGrad; % Return output layer gradient
    case {'purelin', 'softmax'}
        lastHiddenOutput = layerOutputs{numLayers-1};
        lastHiddenGrad = lastHiddenOutput .* (1 - lastHiddenOutput);
        d{numLayers} = error;
        gradient = lastHiddenGrad; % Return last hidden layer gradient
    otherwise
        error('Unsupported output layer activation function');
end

% Backpropagate gradients through hidden layers
for i = numLayers-1:-1:1
    switch net.layer{i}.transferFcn
        case 'logsig'
            layerGrad = layerOutputs{i} .* (1 - layerOutputs{i});
        case 'tansig'
            layerGrad = 1 - layerOutputs{i}.^2;
        case 'elliotsig'
            temp = 1 + abs(layerOutputs{i});
            layerGrad = 1 ./ (temp .* temp);
        otherwise
            error(['Unsupported hidden layer activation function: ', net.layer{i}.transferFcn]);
    end
    
    % Calculate gradient for all layers
    d{i} = layerGrad .* (net.weight{i+1}' * d{i+1});
end

% Calculate weight and bias gradients
dw = cell(numLayers, 1);
db = cell(numLayers, 1);

for i = 1:numLayers
    if i == 1
        % First layer uses network input
        dw{i} = d{i} * input' / batchSize;
    else
        % Other layers use previous layer output
        dw{i} = d{i} * layerOutputs{i-1}' / batchSize;
    end
    db{i} = sum(d{i}, 2) / batchSize;
end

% Update weights and biases using learning rate and momentum
% This uses net.trainParam.lr and net.trainParam.mc
learningRate = net.trainParam.lr;
momentum = net.trainParam.mc;

for i = 1:numLayers
    if i > 1
        % Initialize storage for momentum calculations
        if ~isfield(net, 'prevDeltaW')
            net.prevDeltaW = cell(numLayers, 1);
            net.prevDeltaB = cell(numLayers, 1);
            for j = 1:numLayers
                net.prevDeltaW{j} = zeros(size(net.weight{j}));
                net.prevDeltaB{j} = zeros(size(net.bias{j}));
            end
        end
        
        % Calculate weight and bias updates with momentum
        deltaW = learningRate * dw{i} + momentum * net.prevDeltaW{i};
        deltaB = learningRate * db{i} + momentum * net.prevDeltaB{i};
        
        % Update weights and biases
        net.weight{i} = net.weight{i} + deltaW;
        net.bias{i} = net.bias{i} + deltaB;
        
        % Save current updates for next momentum calculation
        net.prevDeltaW{i} = deltaW;
        net.prevDeltaB{i} = deltaB;
    end
end

% Calculate average gradient magnitude as return value
if isnumeric(gradient)
    gradient = mean(mean(abs(gradient)));
end
end