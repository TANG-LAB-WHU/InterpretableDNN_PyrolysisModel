function perf = nneval(net, input, target)
% NNEVAL Evaluates neural network performance on a given dataset
% Inputs:
%   net - Neural network structure
%   input - Input data
%   target - Target data
% Outputs:
%   perf - Performance metric (MSE)

% Use updated nnff function to calculate network output
[~, output] = nnff(net, input);

% Calculate MSE performance
error = target - output;
perf = mean(mean(error.^2));
end