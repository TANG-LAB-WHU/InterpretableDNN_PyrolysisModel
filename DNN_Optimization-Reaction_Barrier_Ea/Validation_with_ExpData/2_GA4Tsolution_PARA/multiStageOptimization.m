function [best_X, best_fval] = multiStageOptimization_improved(Ea, beta, G_alpha_target, alpha_value, lb, ub)
    % This function performs multi-stage optimization to improve fitting results
    % Inputs:
    %   Ea: activation energy in J/mol
    %   beta: heating rate in K/min
    %   G_alpha_target: target G(alpha) value
    %   alpha_value: current conversion value
    %   lb: lower bound for optimization variables [T_min, A_min]
    %   ub: upper bound for optimization variables [T_max, A_max]
    
    % Number of random starting points
    numStartPoints = 3;
    best_fval = Inf;
    best_X = [];
    
    % Check inputs for NaN or inf values
    if isnan(G_alpha_target) || isinf(G_alpha_target)
        fprintf('Warning: Invalid G_alpha_target value detected: %f\n', G_alpha_target);
        % Return default values if input is invalid
        best_X = [500, 1e20]; % Default reasonable values
        best_fval = 1e10;     % High error to indicate problem
        return;
    end
    
    % Define fitness function with weighting based on conversion
    fitnessFcn = @(X) weightedFitness(X, Ea, beta, G_alpha_target, alpha_value);
    
    % Stage 1: Genetic Algorithm with different settings
    % GA settings
    nvars = 2;
    A = [];
    b = [];
    Aeq = [];
    beq = [];
    nonlcon = [];
    
    % Safety check for bounds
    if any(isnan(lb)) || any(isinf(lb)) || any(isnan(ub)) || any(isinf(ub))
        fprintf('Warning: Invalid bounds detected for alpha=%f. Using default bounds.\n', alpha_value);
        lb = [300, 1e15];
        ub = [900, 1e25];
    end
    
    try
        % Standard GA settings - balanced approach
        options_ga = optimoptions('ga');
        options_ga.PopulationSize = 150;
        options_ga.MaxGenerations = 300; 
        options_ga.FunctionTolerance = 1e-35;
        options_ga.ConstraintTolerance = 1e-5;
        options_ga.SelectionFcn = 'selectiontournament';
        options_ga.CrossoverFraction = 0.8;
        options_ga.Display = 'off';
        
        % Run GA optimization with try/catch
        [X_ga, fval_ga] = ga(fitnessFcn, nvars, A, b, Aeq, beq, lb, ub, nonlcon, options_ga);
        
        % Update best solution
        if fval_ga < best_fval
            best_fval = fval_ga;
            best_X = X_ga;
        end
        
        % Stage 2: Pattern Search for local refinement
        options_ps = optimoptions('patternsearch');
        options_ps.Display = 'off';
        options_ps.MaxIterations = 200;
        options_ps.MaxFunctionEvaluations = 2000;
        options_ps.FunctionTolerance = 1e-35;
        
        % Run pattern search from GA result
        [X_ps, fval_ps] = patternsearch(fitnessFcn, X_ga, A, b, Aeq, beq, lb, ub, nonlcon, options_ps);
        
        % Update best solution
        if fval_ps < best_fval
            best_fval = fval_ps;
            best_X = X_ps;
        end
        
        % Stage 3: fmincon for fine-tuning
        options_fmincon = optimoptions('fmincon');
        options_fmincon.Display = 'off';
        options_fmincon.MaxIterations = 200;
        options_fmincon.MaxFunctionEvaluations = 2000;
        options_fmincon.OptimalityTolerance = 1e-35;
        
        % Run fmincon from current best
        [X_fmincon, fval_fmincon] = fmincon(fitnessFcn, best_X, A, b, Aeq, beq, lb, ub, nonlcon, options_fmincon);
        
        % Update best solution
        if fval_fmincon < best_fval
            best_fval = fval_fmincon;
            best_X = X_fmincon;
        end
    catch ME
        % Handle optimization errors
        fprintf('Optimization error: %s\n', ME.message);
        if isempty(best_X)
            % If no solution found, return midpoint of bounds as fallback
            best_X = [(lb(1) + ub(1))/2, (lb(2) + ub(2))/2];
            best_fval = fitnessFcn(best_X); % Compute actual error for this fallback
        end
    end
    
    % Validate final solution
    if isempty(best_X) || any(isnan(best_X)) || any(isinf(best_X))
        fprintf('Warning: Invalid optimization result. Using fallback values.\n');
        best_X = [(lb(1) + ub(1))/2, (lb(2) + ub(2))/2];
        best_fval = 1e10; % High error to indicate problem
    end
end
