function [lb, ub] = generateAdaptiveBounds_improved(j, alpha_value, base_lb)
    % This function generates adaptive bounds for optimization based on 
    % conversion level and reaction stage, with improved error handling
    % Inputs:
    %   j: index of the activation energy being processed
    %   alpha_value: conversion value
    %   base_lb: basic lower bounds from the original script
    
    % Default bounds (fallback values)
    default_T_lb = 300;
    default_A_lb = 1e15;
    default_T_ub = 973;
    default_A_ub = 1e25;
    
    % Extract base bound values - safely check indices
    if j <= 0 || isempty(base_lb)
        % Use defaults if j is invalid or base_lb is empty
        base_T_lb = default_T_lb;
        base_A_lb = default_A_lb;
    else
        % Make sure to stay within matrix bounds
        row_idx = min(j, size(base_lb, 1));
        
        % Check if base_lb has valid values
        if isnan(base_lb(row_idx, 1)) || isnan(base_lb(row_idx, 2))
            base_T_lb = default_T_lb;
            base_A_lb = default_A_lb;
        else
            base_T_lb = base_lb(row_idx, 1);
            base_A_lb = base_lb(row_idx, 2);
        end
    end
    
    % Default upper bounds
    T_ub = default_T_ub;
    A_ub = default_A_ub;
    
    % Validate alpha_value
    if isnan(alpha_value) || alpha_value < 0 || alpha_value > 1
        fprintf('Warning: Invalid alpha value: %f. Using default bounds.\n', alpha_value);
        lb = [default_T_lb, default_A_lb];
        ub = [default_T_ub, default_A_ub];
        return;
    end
    
    % Adjust bounds based on conversion range
    if alpha_value < 0.3  % Low conversion (early stage)
        T_lb = max(293, base_T_lb - 20);  % Allow slightly lower temperatures
        A_lb = max(1e14, base_A_lb * 0.5); % Allow lower pre-exponential factors
        
    elseif alpha_value >= 0.3 && alpha_value < 0.7  % Mid conversion
        T_lb = base_T_lb;
        A_lb = base_A_lb;
        
    else  % High conversion (late stage)
        T_lb = max(320, base_T_lb + 10);  % Force slightly higher temperatures 
        A_lb = max(1e14, base_A_lb * 0.8); % Allow flexibility in pre-exponential factor
        A_ub = default_A_ub; % Allow higher upper bound for pre-exponential factor
    end
    
    % Final validation to ensure bounds are valid
    if isnan(T_lb) || isinf(T_lb) || T_lb <= 0
        T_lb = default_T_lb;
    end
    
    if isnan(A_lb) || isinf(A_lb) || A_lb <= 0
        A_lb = default_A_lb;
    end
    
    if isnan(T_ub) || isinf(T_ub) || T_ub <= T_lb
        T_ub = max(T_lb * 1.5, default_T_ub);
    end
    
    if isnan(A_ub) || isinf(A_ub) || A_ub <= A_lb
        A_ub = max(A_lb * 10, default_A_ub);
    end
    
    % Return bounds
    lb = [T_lb, A_lb];
    ub = [T_ub, A_ub];
    
    % Output message for debugging
    fprintf('Bounds for alpha=%.2f: T=[%.1f, %.1f], A=[%.2e, %.2e]\n', alpha_value, T_lb, T_ub, A_lb, A_ub);
end
