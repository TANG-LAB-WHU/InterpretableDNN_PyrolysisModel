function output = nnpredict(net, input)
% NNPREDICT Uses neural network to predict outputs for given inputs
% Inputs:
%   net - Neural network structure
%   input - Input data, matrix size [input_dim x num_samples]
% Outputs:
%   output - Neural network predicted output, matrix size [output_dim x num_samples]

% Validate input dimensions
if ~isequal(size(input, 1), size(net.weight{1}, 2) - 1)
    error('Input dimension mismatch: expected %d, got %d', size(net.weight{1}, 2) - 1, size(input, 1));
end

% Call updated nnff function to calculate network output
[~, output] = nnff(net, input);

end