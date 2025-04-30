function [error, output] = nneval(net, input, target)
% NNEVAL determines loss on a given dataset.

global PS TS

% Check for empty input or target and ensure we have the right number of output arguments
if isempty(input) || isempty(target)
    error = NaN;
    output = [];
    return;
end

% normalization for input and target.
input = nnpreprocess(net.processFcn, input, PS);
target = nnpreprocess(net.processFcn, target, TS);

% Calculate normalization output.
try
    [outLayer, ~, ~] = nnff(net, input, target);
catch ME
    % 如果nnff调用失败，提供默认值
    warning('NeuralNet:EvaluationFailed', '%s', ME.message);
    error = NaN;
    output = [];
    return;
end
Nl = net.numLayer;
output = outLayer{Nl};

% Calculate error and loss using reversed normalization output.
target = nnpostprocess(net.processFcn, target, TS);
output = nnpostprocess(net.processFcn, output, TS);

error = mse(target, output);