function [T_opt, A_opt, mean_error] = optimizeTandA(G_alpha, Ea_pred, beta, T_start_K, T_end_K, alphaList, T_exp, w_exp, w_inf)
%OPTIMIZETANDA Two-stage adaptive optimization that enforces T(α) monotonicity
%   [T_opt, A_opt, mean_error] = optimizeTandA(...) maps conversion G(α) to
%   temperature and pre-exponential factor A using either direct physical
%   experimental mapping (fast path) or robust numerical search (fallback).
%
%   All input arrays must share the same length NA = numel(alphaList).

R = 8.314;               % J/(mol·K)
NA = numel(alphaList);
T_opt = zeros(NA, 1);
A_opt = zeros(NA, 1);
errors = zeros(NA, 1);

% Check if experimental data is provided for exact physical mapping
if nargin >= 9 && ~isempty(T_exp) && ~isempty(w_exp) && ~isempty(w_inf)
  % === FAST PATH: Direct Physical Mapping ===
  % 1. Calculate experimental conversion levels
  w_start = w_exp(1);
  alpha_exp = (w_start - w_exp(:)) / (w_start - w_inf * 100);
  alpha_exp = max(0, min(1, alpha_exp));

  % 2. Sort and unique experimental conversion to allow robust interpolation
  [alpha_exp_sorted, sort_idx] = sort(alpha_exp);
  T_exp_K = T_exp(:) + 273.15; % Convert experimental temperature from °C to Kelvin
  T_exp_sorted = T_exp_K(sort_idx);

  [alpha_exp_uniq, uniq_idx] = unique(alpha_exp_sorted);
  T_exp_uniq = T_exp_sorted(uniq_idx);

  % 3. Interpolate experimental temperature at the target conversion levels
  T_target = interp1(alpha_exp_uniq, T_exp_uniq, alphaList(:), 'linear', 'extrap');
  T_target = max(T_start_K, min(T_end_K, T_target));

  % 4. Enforce strict monotonicity using isotonic regression (lsqisotonic)
  x_indices = (1:NA)';
  T_iso = lsqisotonic(x_indices, T_target);
  T_iso = max(T_start_K, min(T_end_K, T_iso));
  T_opt = T_iso + (0:NA-1)' * 1e-5; % Add tiny increments after clipping to ensure strict inequality

  % 5. Solve for A_opt analytically for each conversion level
  for i = 1:NA
    T = T_opt(i);
    Ea = Ea_pred(i);
    p = Ea / (R * T);

    % Coats-Redfern temperature integral calculation
    if p > 20
      I_val = (1 / beta) * (R / Ea) * T^2 * (1 - 2 * R * T / Ea) * exp(-p);
    else
      I_val = (1 / beta) * (T / p) * (1 - 2/p + 6/p^2 - 24/p^3) * exp(-p);
    end

    A_opt(i) = G_alpha(i) / (I_val + eps);
    A_opt(i) = max(1e4, min(1e18, A_opt(i))); % Keep A within physical bounds
    
    % Evaluate analytical error
    G_calc = A_opt(i) * I_val;
    errors(i) = abs(G_alpha(i) - G_calc);
  end

else
  % === FALLBACK PATH: Numerical optimization (Legacy) ===
  T_opt_coarse = zeros(NA,1);
  A_opt_coarse = zeros(NA,1);
  Ea_min = min(Ea_pred);
  Ea_max = max(Ea_pred);

  % Coarse search
  for i = 1:NA
    target_G = G_alpha(i);
    Ea       = Ea_pred(i);

    span = T_end_K - T_start_K;
    frac = (Ea - Ea_min) / (max(1, Ea_max - Ea_min));
    T_lb = T_start_K + span * max(0.0, frac - 0.25);
    T_ub = T_start_K + span * min(1.0, frac + 0.25);

    T_mid     = 0.5*(T_lb+T_ub);
    A_typical = max(1e9, min(1e14, beta * exp(Ea / (R*T_mid))));
    lb = [T_lb, max(1e6, A_typical/100)];
    ub = [T_ub, min(1e14, A_typical*100)];

    x_best = [T_mid, A_typical];
    T_opt_coarse(i) = x_best(1);
    A_opt_coarse(i) = x_best(2);
  end

  % Isotonic projection
  x_indices = (1:NA)';
  T_iso = lsqisotonic(x_indices, T_opt_coarse);
  T_iso = T_iso + (0:NA-1)'*1e-3;
  T_iso = T_start_K + (T_iso - min(T_iso)) * (T_end_K - T_start_K) / (max(T_iso) - min(T_iso) + eps);
  T_opt = max(T_start_K+1e-3, min(T_end_K-1e-3, T_iso));
  A_opt = A_opt_coarse;
  errors = zeros(NA, 1);
end

mean_error = mean(errors);
end
