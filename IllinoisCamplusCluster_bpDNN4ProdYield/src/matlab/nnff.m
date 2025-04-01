function [layerOutputs, networkOutput, error, loss] = nnff(net, input, target)
% NNFF Neural network forward propagation algorithm
% Inputs:
%   net - Neural network structure
%   input - Input data
%   target - Target data (optional)
% Outputs:
%   layerOutputs - Cell array of outputs from each layer
%   networkOutput - Final network output
%   error - Error (if target is provided)
%   loss - Loss value (if target is provided)

% Check input parameters
hasTarget = (nargin >= 3 && ~isempty(target));

% Get batch size
batchSize = size(input, 2); 

% Get number of network layers
numLayers = length(net.layer);

% Initialize layer output storage
layerOutputs = cell(numLayers, 1);

% Current layer input is the network input
currentInput = input;

% Calculate output for each layer
for i = 1:numLayers
    % Combine weights and biases
    Wb = [net.weight{i}, net.bias{i}];
    
    % Add bias term
    X = [currentInput; ones(1, batchSize)];
    
    % Calculate output based on transfer function
    switch net.layer{i}.transferFcn
        case 'logsig'
            layerOutputs{i} = logsig(Wb * X);
        case 'tansig'
            layerOutputs{i} = tansig(Wb * X);
        case 'elliotsig'
            layerOutputs{i} = elliotsig(Wb * X);
        case 'purelin'
            layerOutputs{i} = purelin(Wb * X);
        case 'softmax'
            layerOutputs{i} = softmax(Wb * X);
        otherwise
            error(['Unsupported transfer function: ', net.layer{i}.transferFcn]);
    end
    
    % Current layer output becomes next layer input
    currentInput = layerOutputs{i};
end

% Network output is the last layer's output
networkOutput = layerOutputs{numLayers};

% If target data is provided, calculate error and loss
if hasTarget
    error = target - networkOutput;
    
    % Calculate loss based on output layer transfer function
    switch net.layer{numLayers}.transferFcn
        case {'logsig', 'purelin', 'tansig', 'elliotsig'}
            loss = 0.5 * sum(sum(error.^2)) / batchSize; % MSE
        case 'softmax'
            % Cross-entropy loss
            loss = -sum(sum(target .* log(networkOutput))) / batchSize;
        otherwise
            error(['Unsupported output layer transfer function: ', net.layer{numLayers}.transferFcn]);
    end
else
    % If no target data provided, return empty error and loss
    error = [];
    loss = [];
end