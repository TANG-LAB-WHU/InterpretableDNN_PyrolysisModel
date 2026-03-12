function error = weightedFitness(X, Ea, beta, G_alpha_target, alpha_value)
    % This function calculates the weighted fitness for optimization
    % Inputs:
    %   X: optimization variables [T, A]
    %   Ea: activation energy in J/mol
    %   beta: heating rate in K/min
    %   G_alpha_target: target value of conversion function G(alpha)
    %   alpha_value: current conversion value
    
    % Calculate basic error
    basic_error = (Integral4Tintegral(X, Ea, beta) - G_alpha_target)^2;
    
    % Apply weights based on conversion ranges
    % Middle conversion ranges are more important for overall curve fitting
    if alpha_value < 0.2
        weight = 1.5;  % Higher weight for initial conversion
    elseif alpha_value > 0.3 && alpha_value < 0.7
        weight = 2.0;  % Highest weight for middle conversion range
    elseif alpha_value > 0.85
        weight = 1.5;  % Higher weight for final conversion
    else
        weight = 1.0;  % Normal weight for other ranges
    end
    
    error = basic_error * weight;
end
