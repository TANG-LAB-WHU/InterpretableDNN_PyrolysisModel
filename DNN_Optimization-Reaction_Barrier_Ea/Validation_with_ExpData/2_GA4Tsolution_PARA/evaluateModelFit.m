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
end
