function [w_pred, T_pred] = generateTGCurveFromG(G_alpha, T_opt, A_opt, Ea_pred, beta, alphaList, w_inf, T_exp)
% Generate TG curve using true kinetic integration
% NEW IMPLEMENTATION: Use Arrhenius kinetics with proper integration
% Inputs:
%   G_alpha: G(alpha) values
%   T_opt: optimized temperatures (K)
%   A_opt: optimized pre-exponential factors (1/min)
%   Ea_pred: predicted activation energies (J/mol)
%   beta: heating rate (K/min)
%   alphaList: conversion values
%   w_inf: final char yield (fraction)
% Outputs:
%   w_pred: predicted weight percentages
%   T_pred: predicted temperatures (K)

cfg = performanceConfig();
if isfield(cfg, 'integration') && isfield(cfg.integration, 'nSteps')
  n_points = cfg.integration.nSteps;  % from config
else
  n_points = 600;  % sensible default if config missing
end

% Create fine temperature grid for integration
% Use the EXPERIMENTAL temperature range to ensure full coverage.
% If T_exp is provided, match its range exactly; otherwise fall back to defaults.
if nargin >= 8 && ~isempty(T_exp)
  T_min = min(T_exp(:)) + 273.15;  % Convert experimental min from °C to K
  T_max = max(T_exp(:)) + 273.15;  % Convert experimental max from °C to K
else
  T_min = 293.15;   % Fallback: 20°C
  T_max = 1173.15;  % Fallback: 900°C
end
T_fine = linspace(T_min, T_max, n_points);

% Initialize arrays
alpha_fine = zeros(n_points, 1);
w_fine = zeros(n_points, 1);

% --- Determine f(alpha) strictly from the selected mechanism(s)
% G_alpha encodes the integral of 1/f(alpha) up to each conversion level.
% Recover the differential rate function so that integration follows the
% exact mechanism selected during model comparison.
if nargin < 6 || isempty(alphaList)
  error('alphaList is required to compute mechanism-specific f(alpha)');
end
if numel(G_alpha) ~= numel(alphaList)
  error('G_alpha and alphaList must have the same length');
end

% Numerical derivative of G(alpha)
G_alpha = G_alpha(:);
alphaList = alphaList(:);
dG_dalpha = gradient(G_alpha, alphaList);

R = 8.314;  % Universal gas constant (J/mol/K)

% Safeguard against zero/negative slopes (should not occur for valid G)
small = 1e-12;
dG_dalpha = max(dG_dalpha, small);
f_alpha_vec = 1 ./ dG_dalpha;

alphaVec = double(real(alphaList(:)));
fVals    = double(real(f_alpha_vec(:)));

% Build interpolant function for f(alpha) using interp1 (safer for mixed types)
f_interp = @(a) interp1(alphaVec, fVals, max(0, min(1, a)), 'linear', 'extrap');

% Integrate Arrhenius kinetics for each temperature point
for i = 1:n_points
  T_current = T_fine(i);

  % Interpolate kinetic parameters (prefer linear for speed, fallback to bounds)
  if T_current < min(T_opt)
    Ea_current = Ea_pred(1);
    A_current  = A_opt(1);
  elseif T_current > max(T_opt)
    Ea_current = Ea_pred(end);
    A_current  = A_opt(end);
  else
    Ea_current = interp1(T_opt, Ea_pred, T_current, 'linear', 'extrap');
    A_current  = interp1(T_opt, A_opt,  T_current, 'linear', 'extrap');
  end

  % integrate the rate equation using Runge-Kutta integration for better accuracy
  if i == 1
    alpha_fine(i) = 0;  % Start with α = 0
  else
    dT = T_fine(i) - T_fine(i-1);
    k_prev = A_current * exp(-Ea_current / (R * T_fine(i-1)));

    % employ the Runge-Kutta method to integrate the rate equation
    alpha_prev = max(0, min(1, real(alpha_fine(i-1))));
    f_prev = f_interp(alpha_prev);

    % RK4 integration
    k1 = (k_prev / beta) * f_prev;
    alpha_temp = max(0, min(1, alpha_prev + 0.5*dT*k1));
    k2 = (k_prev / beta) * f_interp(alpha_temp);
    alpha_temp = max(0, min(1, alpha_prev + 0.5*dT*k2));
    k3 = (k_prev / beta) * f_interp(alpha_temp);
    alpha_temp = max(0, min(1, alpha_prev + dT*k3));
    k4 = (k_prev / beta) * f_interp(alpha_temp);

    alpha_fine(i) = alpha_prev + (dT/6) * (k1 + 2*k2 + 2*k3 + k4);
    alpha_fine(i) = max(0, min(1, alpha_fine(i)));

  end

  % Convert α to weight percentage
  w_fine(i) = 100 * (1 - alpha_fine(i) * (1 - w_inf));
end

% Ensure monotonic decrease and apply w_inf constraint
w_fine = cummin(w_fine);
w_fine = max(w_fine, w_inf * 100);
w_fine(end) = w_inf * 100;

% Convert temperature grid from Kelvin to Celsius for comparison with experimental data
T_fine_C = T_fine - 273.15;
T_opt_C  = T_opt  - 273.15;

% Return results on the FULL fine temperature grid (now in °C)
% This ensures the predicted curve covers the entire experimental range
w_pred = w_fine(:)';
T_pred = T_fine_C(:)';

% Final validation
w_pred = cummin(w_pred);
w_pred = max(w_pred, w_inf * 100);
w_pred(end) = w_inf * 100;
end
