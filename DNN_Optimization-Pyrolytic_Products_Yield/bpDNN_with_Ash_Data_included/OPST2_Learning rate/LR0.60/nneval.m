function [error, output] = nneval(net, input, target)
% NNEVAL determines loss on a given dataset.

global PS TS

% normalization for input and target.
input = nnpreprocess(net.processFcn, input, PS);
target = nnpreprocess(net.processFcn, target, TS);

% Calculate normalization output.
outLayer = nnff(net, input, target);
Nl = net.numLayer;
output = outLayer{Nl};

% Calculate error and loss using reversed normalization output.
target = nnpostprocess(net.processFcn, target, TS);
output = nnpostprocess(net.processFcn, output, TS);

error = mse(target, output);