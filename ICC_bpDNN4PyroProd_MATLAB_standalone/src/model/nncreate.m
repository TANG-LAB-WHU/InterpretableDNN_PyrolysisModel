function net = nncreate(numInputs, hiddenLayer, numOutputs, hiddenTF, outputTF)
% NNCREATE forges a neural network with a certain number of weights and
%       biases. All parametes including learning ratio (lr), momoent
%       coefficient (mc) , trasnsfer fucntion(transferFcn),
%       stop criteria (maximal iteration (epoch)), expected performance
%       (goal), minimal gradient (mingrad), number of successive iteration
%       valdation performance failing to decrease (maxfail) ), divide
%       function (divideFcn) and etc, are set up.
%
%   Inputs:
%       numInputs - Number of input features
%       hiddenLayer - A row vector specifying number of neurons in each hidden layer.
%       numOutputs - Number of output targets
%       hiddenTF - Transfer function for hidden layers (e.g., 'tansig', 'logsig', 'poslin')
%       outputTF - Transfer function for output layer (e.g., 'purelin')
%   output:
%           net - The created network with default parameters and initialized weights/biases.

% --- Start Debugging ---
fprintf('DEBUG nncreate: Received numInputs = %d\n', numInputs);
fprintf('DEBUG nncreate: Received hiddenLayer = [%s]\n', num2str(hiddenLayer));
fprintf('DEBUG nncreate: Received numOutputs = %d\n', numOutputs);
if nargin >= 4
    fprintf('DEBUG nncreate: Received hiddenTF = %s\n', hiddenTF);
else
    fprintf('DEBUG nncreate: Received hiddenTF = (defaulting)\n');
end
if nargin >= 5
    fprintf('DEBUG nncreate: Received outputTF = %s\n', outputTF);
else
        fprintf('DEBUG nncreate: Received outputTF = (defaulting)\n');
end
% --- End Debugging ---


% Set defaults for missing arguments
if nargin < 5 || isempty(outputTF) % Check for empty too
    outputTF = 'purelin';
        fprintf('DEBUG nncreate: Using default outputTF = %s\n', outputTF);
end
if nargin < 4 || isempty(hiddenTF) % Check for empty too
    hiddenTF = 'logsig';
        fprintf('DEBUG nncreate: Using default hiddenTF = %s\n', hiddenTF);
end
if nargin < 3 || isempty(numOutputs)
    numOutputs = 1;
        fprintf('DEBUG nncreate: Using default numOutputs = %d\n', numOutputs);
end
if nargin < 2 || isempty(hiddenLayer)
    hiddenLayer = 10; % Default hidden layer size if not provided
        fprintf('DEBUG nncreate: Using default hiddenLayer = [%s]\n', num2str(hiddenLayer));
end
if nargin < 1 || isempty(numInputs)
    numInputs = 1; % Default input size if not provided
        fprintf('DEBUG nncreate: Using default numInputs = %d\n', numInputs);
end

% Initialize network structure
net.numInput = numInputs;
net.numOutput = numOutputs;

% Set up layer structure description
layerSizes = [numInputs, hiddenLayer, numOutputs]; % Include input size for dimension calculation convenience
Nl = length(hiddenLayer) + 1; % Number of layers (Hidden + Output)
net.numLayer = Nl;
net.layer = cell(Nl, 1); % Preallocate cell array

fprintf('DEBUG nncreate: Total number of layers (Nl) = %d\n', Nl);

for i = 1 : Nl
    currentLayerSize = layerSizes(i+1); % Size of the current layer (index shifted by 1 due to input size inclusion)
    net.layer{i}.size = currentLayerSize;
    fprintf('DEBUG nncreate: Layer %d size = %d\n', i, net.layer{i}.size);
    if i == Nl
        net.layer{i}.transferFcn = outputTF;
            fprintf('DEBUG nncreate: Layer %d (Output) transferFcn = %s\n', i, net.layer{i}.transferFcn);
    else
        net.layer{i}.transferFcn = hiddenTF;
            fprintf('DEBUG nncreate: Layer %d (Hidden) transferFcn = %s\n', i, net.layer{i}.transferFcn);
    end
    % Add fields for regularization parameters (can be adjusted later)
    net.layer{i}.rw = 1; % Weight regularization factor (default 1)
    net.layer{i}.rb = 1; % Bias regularization factor (default 1)
end

% --- START: Added Weight and Bias Initialization ---
net.weight = cell(Nl, 1); % Initialize as cell array
net.bias = cell(Nl, 1);   % Initialize as cell array

fprintf('DEBUG nncreate: Initializing weights and biases...\n');
for i = 1 : Nl
    % Layer i connects from layer i-1 (or inputs for i=1)
    % Size of the previous layer (or input layer)
    previousLayerSize = layerSizes(i); % Index i matches the layerSizes array directly
    % Size of the current layer
    currentLayerSize = layerSizes(i+1); % Index i+1 matches the layerSizes array directly

    % Calculate dimensions
    weight_rows = currentLayerSize;
    weight_cols = previousLayerSize;
    bias_rows = currentLayerSize;
    bias_cols = 1;

    fprintf('DEBUG nncreate: Layer %d: Previous Size=%d, Current Size=%d\n', i, previousLayerSize, currentLayerSize);
    fprintf('DEBUG nncreate: Calculated Layer %d weight size = [%d, %d]\n', i, weight_rows, weight_cols);
    fprintf('DEBUG nncreate: Calculated Layer %d bias size = [%d, %d]\n', i, bias_rows, bias_cols);

    % Initialize weights with small random numbers (randn is standard normal distribution)
    % Scaling factor like 0.01 helps prevent saturation in sigmoid functions initially
    net.weight{i} = randn(weight_rows, weight_cols) * 0.01;
    % Initialize biases to zero
    net.bias{i} = zeros(bias_rows, bias_cols);

    fprintf('DEBUG nncreate: Initialized net.weight{%d} and net.bias{%d}\n', i, i);
end
% --- END: Added Weight and Bias Initialization ---


% --- Default Parameter Settings ---
% Normalization
net.processFcn = 'mapminmax'; % 'mapminmax' or 'mapstd'
net.processParam.ymax = 1;
net.processParam.ymin = -1;

% Data set division
net.divideFcn = 'dividerand';
net.divideParam.trainRatio = 1; % Default to using all data for training initially
net.divideParam.valRatio = 0;
net.divideParam.testRatio = 0;

% Training parameters (these are often overridden by train_best_model.m)
net.trainParam.mc = 0.90;
net.trainParam.lr = 0.01;
net.trainParam.lr_inc = 1.05;
net.trainParam.lr_dec = 0.70;
net.trainParam.goal = 1e-6;
net.trainParam.epoch = 400;
net.trainParam.min_grad = 1e-5;
net.trainParam.max_fail = 6;

net.performFcn = 'mse';
net.adaptFcn = 'none';

% Display
net.trainParam.showWindow = false; % Keep false for batch jobs
net.trainParam.showCommandLine = false; % Keep false for batch jobs

fprintf('DEBUG nncreate: Network creation complete.\n');

end