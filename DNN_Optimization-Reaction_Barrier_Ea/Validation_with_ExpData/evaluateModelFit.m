function [error, r2] = evaluateModelFit(predicted, experimental)
    % This function calculates error metrics for model fit evaluation
    % Inputs:
    %   predicted: predicted values from the model
    %   experimental: experimental values
    % Outputs:
    %   error: root mean square error
    %   r2: coefficient of determination (R-squared)
    
    % Calculate root mean square error
    error = sqrt(mean((predicted - experimental).^2));
    
    % Calculate coefficient of determination (R-squared)
    SST = sum((experimental - mean(experimental)).^2);
    SSE = sum((experimental - predicted).^2);
    r2 = 1 - SSE/SST;
    
    % Handle any potential numerical issues
    if isnan(r2) || isinf(r2)
        r2 = 0; % Default value for invalid calculations
    end
    
    % R-squared should not exceed 1 or be less than 0 in proper models
    r2 = max(0, min(1, r2));
end
