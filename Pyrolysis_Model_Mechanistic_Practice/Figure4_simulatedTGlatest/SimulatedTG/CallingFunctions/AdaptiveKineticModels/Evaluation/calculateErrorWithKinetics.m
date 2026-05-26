function error = calculateErrorWithKinetics(w_pred, T_pred, w_exp, T_exp, w_inf, G_alpha, T_opt, A_opt, Ea_pred, beta, alphaList)
% Calculate error between predicted and experimental TG curves using kinetic model integration
% NEW IMPLEMENTATION: Use kinetic model integration instead of interpolation

% --------------------------------------------------
% 1. Use kinetic model integration for all experimental temperature points
%    with optional fast path (direct interpolation)
% --------------------------------------------------
cfg = performanceConfig();

% Fast path: interpolate predicted curve onto experimental temperatures
% T_pred is in °C; T_exp also in °C
w_interp = interp1(T_pred(:), w_pred(:), T_exp(:), 'linear', 'extrap');
w_interp = w_interp(:);

% --------------------------------------------------
% 2. Normalized error calculation (percentage-based)
% --------------------------------------------------
% Calculate relative error instead of absolute
relative_error = abs(w_interp - w_exp(:)) ./ (w_exp(:) + 1e-6);  % Avoid division by zero

% --------------------------------------------------
% 3. Temperature-weighted error
% --------------------------------------------------
cfg = performanceConfig();

% Check if adaptive weighting is enabled (default: true)
useAdaptiveWeights = true;
if isfield(cfg,'error') && isfield(cfg.error,'useAdaptiveWeights')
  useAdaptiveWeights = cfg.error.useAdaptiveWeights;
end

if useAdaptiveWeights
  % Calculate adaptive weights from experimental DTG curve (no human assumptions)
  weights = calculateAdaptiveWeights(T_exp, w_exp);
else
  % Fallback: uniform weights for zero-assumption physically rigorous fitting
  weights = ones(size(T_exp));
end

% Calculate weighted error
weighted_error = sqrt(mean(weights .* relative_error.^2)) * 100;

% --------------------------------------------------
% 4. Shape penalty (ensure proper TG curve shape)
% --------------------------------------------------
% Check if curve is monotonically decreasing
if any(diff(w_interp) > 0.1)  % Allow small numerical noise
  shape_penalty = 50;
else
  shape_penalty = 0;
end

% --------------------------------------------------
% 5. Temperature range penalty
% --------------------------------------------------
temp_range = max(T_pred) - min(T_pred);
expected_range = 900 - 20;  % Dataset range: 20-900°C
if temp_range < expected_range * 0.8  % Too narrow temperature range
  range_penalty = 30;
elseif temp_range > expected_range * 1.2  % Too wide temperature range
  range_penalty = 20;
else
  range_penalty = 0;
end

% --------------------------------------------------
% 6. Endpoint penalty
% --------------------------------------------------
endpoint_penalty = 0;
endpoint_error = abs(w_interp(end) - w_inf * 100);
if endpoint_error > 2.0  % More than 2% deviation
  endpoint_penalty = endpoint_error * 5;
end

% 6b. Optional alignment penalties
cfg = performanceConfig();
t50_penalty = 0; dtg_penalty = 0;
if isfield(cfg,'error') && isfield(cfg.error,'enableT50Penalty') && cfg.error.enableT50Penalty
  w50 = 100 * (1 - 0.5 * (1 - w_inf));  % weight at alpha = 0.5
  % robust interpolation using flipped arrays (w strictly decreasing expected)
  try
    T50_exp  = interp1(flip(w_exp),  flip(T_exp),  w50, 'linear', 'extrap');
    T50_pred = interp1(flip(w_interp),flip(T_exp),  w50, 'linear', 'extrap');
    if isfinite(T50_exp) && isfinite(T50_pred)
      t50_penalty = cfg.error.t50Weight * abs(T50_pred - T50_exp);
    end
  catch
    % ignore interpolation errors
  end
end
if isfield(cfg,'error') && isfield(cfg.error,'enableDTGPenalty') && cfg.error.enableDTGPenalty
  dT  = max(gradient(T_exp), 1e-6);
  dtg_exp  = -gradient(w_exp)./dT;
  dtg_pred = -gradient(w_interp)./dT;
  % EXPANDED FOCUS: Evaluate DTG over the entire chemical pyrolysis range (150-900°C)
  % This ensures the slow, continuous lignin degradation rate is matched.
  focus = (T_exp >= 150) & (T_exp <= 900);
  if any(focus)
    dtg_penalty = cfg.error.dtgWeight * sqrt(mean((dtg_pred(focus) - dtg_exp(focus)).^2));
  end
end

% --------------------------------------------------
% 7. Combine all error components
% --------------------------------------------------
error = weighted_error + shape_penalty + range_penalty + endpoint_penalty + t50_penalty + dtg_penalty;

% Ensure error is reasonable (not too large or too small)
error = max(0.1, min(error, 1000));
end

function w = calculateKineticWeight(T_target, G_alpha, T_opt, A_opt, Ea_pred, beta, alphaList, w_inf)
% Calculate weight at specific temperature using kinetic model integration

% Handle scalar vs vector parameters
if isscalar(Ea_pred)
  Ea_current = Ea_pred;
else
  % Find corresponding kinetic parameters (by temperature interpolation)
  [~, idx] = min(abs(T_opt - T_target));
  Ea_current = Ea_pred(idx);
end

if isscalar(A_opt)
  A_current = A_opt;
else
  % Find corresponding kinetic parameters (by temperature interpolation)
  [~, idx] = min(abs(T_opt - T_target));
  A_current = A_opt(idx);
end

% Recover f(alpha) function from G(alpha)
dG_dalpha = gradient(G_alpha, alphaList);
small = 1e-12;
dG_dalpha = max(dG_dalpha, small);
f_alpha_vec = 1 ./ dG_dalpha;

alphaVec = double(real(alphaList(:)));
fVals = double(real(f_alpha_vec(:)));
f_interp = @(a) interp1(alphaVec, fVals, max(0, min(1, a)), 'linear', 'extrap');

% Integrate to target temperature (configurable steps)
cfg = performanceConfig();
if isfield(cfg,'integration') && isfield(cfg.integration,'nSteps')
  n_steps = cfg.integration.nSteps;
else
  n_steps = 200;
end
alpha_current = integrateToTemperature(T_target, A_current, Ea_current, beta, f_interp, n_steps);

% Convert to weight percentage
w = 100 * (1 - alpha_current * (1 - w_inf));
end

function alpha = integrateToTemperature(T_target, A, Ea, beta, f_interp, n_steps)
% Integrate kinetic equation to reach target temperature

R = 8.314;  % Universal gas constant (J/mol/K)

T_start = 293.15;  % Starting temperature (20°C)
dT = (T_target - T_start) / n_steps;

alpha = 0;  % Starting conversion level

for i = 1:n_steps
  T_current = T_start + i * dT;
  k = A * exp(-Ea / (R * T_current));

  % RK4 integration
  f_alpha = f_interp(alpha);
  k1 = (k / beta) * f_alpha;

  alpha_temp = max(0, min(1, alpha + 0.5*dT*k1));
  f_temp = f_interp(alpha_temp);
  k2 = (k / beta) * f_temp;

  alpha_temp = max(0, min(1, alpha + 0.5*dT*k2));
  f_temp = f_interp(alpha_temp);
  k3 = (k / beta) * f_temp;

  alpha_temp = max(0, min(1, alpha + dT*k3));
  f_temp = f_interp(alpha_temp);
  k4 = (k / beta) * f_temp;

  alpha = alpha + (dT/6) * (k1 + 2*k2 + 2*k3 + k4);
  alpha = max(0, min(1, alpha));
end
end

function weights = calculateAdaptiveWeights(T_exp, w_exp)
% Calculate adaptive error weights based on experimental DTG curve
% This function automatically detects decomposition stages from the data,
% enabling universal applicability across different biomass types.
%
% The weight distribution is driven by:
%   1. DTG magnitude (higher mass loss rate -> higher weight)
%   2. Peak detection (regions around DTG peaks get extra weight)
%
% Input:
%   T_exp - Experimental temperature array (°C)
%   w_exp - Experimental weight array (%)
%
% Output:
%   weights - Adaptive weight array (same size as T_exp)

% Ensure column vectors
T_exp = T_exp(:);
w_exp = w_exp(:);

% --------------------------------------------------
% 1. Calculate DTG (derivative thermogravimetry)
% --------------------------------------------------
dT = gradient(T_exp);
dT = max(abs(dT), 1e-6);  % Avoid division by zero
dtg = -gradient(w_exp) ./ dT;  % Mass loss rate (positive values)

% --------------------------------------------------
% 2. Smooth DTG to reduce noise
% --------------------------------------------------
% Use moving average with window size proportional to data length
windowSize = max(5, round(length(dtg) / 50));
dtg_smooth = movmean(dtg, windowSize);

% --------------------------------------------------
% 3. Base weights from normalized DTG
% --------------------------------------------------
dtg_max = max(abs(dtg_smooth));
if dtg_max > 1e-6
  dtg_norm = abs(dtg_smooth) / dtg_max;
else
  dtg_norm = zeros(size(dtg_smooth));
end

% Base weight: scale from 1 to 6 based on DTG magnitude
weights = 1 + 5 * dtg_norm;

% --------------------------------------------------
% 4. Detect peaks and add extra weight around them
% --------------------------------------------------
try
  % Find significant peaks in the DTG curve
  minPeakProminence = 0.1 * dtg_max;
  minPeakDistance = round(length(dtg_smooth) / 20);
  
  [~, peakLocs] = findpeaks(dtg_smooth, ...
    'MinPeakProminence', minPeakProminence, ...
    'MinPeakDistance', minPeakDistance);
  
  % Add extra weight around each peak (±30°C window)
  for i = 1:length(peakLocs)
    peakTemp = T_exp(peakLocs(i));
    peakRegion = abs(T_exp - peakTemp) < 30;
    weights(peakRegion) = weights(peakRegion) * 1.3;
  end
catch
  % If peak detection fails, continue with base weights
end

% --------------------------------------------------
% 5. Apply minimum weight for low-activity regions
% --------------------------------------------------
% Low weight for moisture region (<150°C)
lowActivity = T_exp < 150 | dtg_norm < 0.05;
weights(lowActivity) = max(weights(lowActivity) * 0.3, 0.5);

% LIGNIN TAIL BOOST: High-temperature region (>450°C)
% Instead of suppressing it, we enforce a minimum weight to ensure GA respects lignin degradation
highTempRegion = T_exp > 450;
weights(highTempRegion) = max(weights(highTempRegion), 2.5);

% --------------------------------------------------
% 6. Normalize and bound weights
% --------------------------------------------------
weights = max(0.5, min(weights, 10));

end