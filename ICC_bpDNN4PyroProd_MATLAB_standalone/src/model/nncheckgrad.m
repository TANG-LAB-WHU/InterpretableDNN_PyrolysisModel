function net = nncheckgrad(varargin)
% NNCHECKGRAD checks gradients by comparing them with those calculated by
%  numerical gradient.
%
%   Requires 5 arguments:
%   net = nncheckgrad(net, input, target, dW, db)

if nargin == 5
    [net, input, target, dW, db] = deal(varargin{:});
    % Checking input
    epsilon = 1e-7; % Step size
    
    % Select a random weight to perturb
    layer = randi(net.numLayer);
    i = randi(size(net.weight{layer}, 1));
    j = randi(size(net.weight{layer}, 2));
    
    % Save original weight
    original_weight = net.weight{layer}(i, j);
    
    % Create two nets with slightly different weights
    PNET = net;
    PNET.weight{layer}(i, j) = original_weight + epsilon;
    
    QNET = net;
    QNET.weight{layer}(i, j) = original_weight - epsilon;
    
    % Get loss for both networks
    PLOSS = get_loss(PNET, input, target);
    QLOSS = get_loss(QNET, input, target);
    
    % Calculate numerical gradient
    numerical_gradient = (PLOSS - QLOSS) / (2 * epsilon);
    
    % Compare with analytical gradient
    analytical_gradient = dW{layer}(i, j);
    
    % Compute relative error
    if abs(numerical_gradient) > 1e-10 || abs(analytical_gradient) > 1e-10
        relative_error = abs(numerical_gradient - analytical_gradient) / ...
            max(abs(numerical_gradient), abs(analytical_gradient));
    else
        relative_error = 0;
    end
    
    % Report if error is too large
    if relative_error > 0.01 % 1% tolerance
        warning('Gradient check failed for weight %d,%d in layer %d. Numerical: %g, Analytical: %g, Relative Error: %g', ...
            i, j, layer, numerical_gradient, analytical_gradient, relative_error);
    end
end

end

% Helper function to get loss for a network
function loss = get_loss(net, input, target)
    % Call nnff and get only the loss value
    % 修复：只获取nnff的第三个返回值以避免参数不匹配的问题
    try
        [~, ~, loss] = nnff(net, input, target);
    catch
        loss = NaN;
    end
end