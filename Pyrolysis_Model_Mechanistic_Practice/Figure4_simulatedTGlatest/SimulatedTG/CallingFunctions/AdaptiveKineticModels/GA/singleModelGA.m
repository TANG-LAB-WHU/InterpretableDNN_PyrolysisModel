function [bestParams, bestError] = singleModelGA(modelCategory, alphaList, Ea_pred, beta, T_start_K, T_end_K, T_exp, w_exp, w_inf)
% Unified single model GA optimizer with Two-Stage Coarse-to-Fine strategy
%
% Input parameters:
%   modelCategory - Model category
%   alphaList - Conversion level list
%   Ea_pred - Predicted activation energy
%   beta - Heating rate
%   T_start_K - Start temperature (K)
%   T_end_K - End temperature (K)
%   T_exp - Experimental temperature data
%   w_exp - Experimental weight data
%   w_inf - Final residue
%
% Output parameters:
%   bestParams - Best parameters
%   bestError - Best error

cfg = performanceConfig();
allModels = getAllModelsInCategory(modelCategory);
twoStageEnabled = isfield(cfg, 'twoStageGA') && cfg.twoStageGA.enabled;

fprintf('Testing %s category with %d models using GA optimization...\n', modelCategory, length(allModels));

if twoStageEnabled
  topN = cfg.twoStageGA.topN;

  % === STAGE 1: Coarse GA Screening ===
  fprintf('  [Stage 1] Coarse GA screening for %d models...\n', length(allModels));
  screeningResults = cell(length(allModels), 1);

  for i = 1:length(allModels)
    modelType = allModels{i};

    if needsParameters(modelType)
      % Use coarse GA for quick screening
      [params, error] = optimizeSingleModelWithGA(modelType, alphaList, Ea_pred, beta, T_start_K, T_end_K, T_exp, w_exp, w_inf, 'coarse');
    else
      % No parameter model, test directly
      params = struct();
      [error, params] = testSingleModel(modelType, params, alphaList, Ea_pred, beta, T_start_K, T_end_K, T_exp, w_exp, w_inf);
    end

    screeningResults{i} = struct('modelType', modelType, 'error', error, 'params', params);
    fprintf('    %s: %.4f%%\n', modelType, error);
  end

  % Rank and select Top-N
  errors = cellfun(@(x) x.error, screeningResults);
  [sortedErrors, sortIdx] = sort(errors);
  topN = min(topN, length(allModels));

  fprintf('  [Stage 2] Fine GA optimization for Top-%d candidates:\n', topN);
  for rank = 1:topN
    idx = sortIdx(rank);
    modelType = screeningResults{idx}.modelType;
    fprintf('    #%d: %s (coarse error: %.4f%%)\n', rank, modelType, sortedErrors(rank));

    if needsParameters(modelType)
      % Use fine GA for precise optimization
      [params, error] = optimizeSingleModelWithGA(modelType, alphaList, Ea_pred, beta, T_start_K, T_end_K, T_exp, w_exp, w_inf, 'fine');
      screeningResults{idx}.params = params;
      screeningResults{idx}.error = error;
      fprintf('        -> fine error: %.4f%%\n', error);
    end
  end

  % Find best after Stage 2
  errors = cellfun(@(x) x.error, screeningResults);
  [bestError, bestIdx] = min(errors);
  bestParams = screeningResults{bestIdx}.params;
  bestParams.modelType = screeningResults{bestIdx}.modelType;

else
  % === Original single-stage logic (fallback) ===
  bestError = Inf;
  bestParams = struct();

  for i = 1:length(allModels)
    modelType = allModels{i};
    fprintf('  Testing model: %s\n', modelType);

    if needsParameters(modelType)
      [params, error] = optimizeSingleModelWithGA(modelType, alphaList, Ea_pred, beta, T_start_K, T_end_K, T_exp, w_exp, w_inf, 'single_model');
    else
      params = struct();
      [error, params] = testSingleModel(modelType, params, alphaList, Ea_pred, beta, T_start_K, T_end_K, T_exp, w_exp, w_inf);
    end

    if error < bestError
      bestError = error;
      bestParams = params;
      bestParams.modelType = modelType;
      fprintf('    New best: %s (error: %.4f%%)\n', modelType, error);
    end
  end
end

if isfield(bestParams, 'modelType')
  fprintf('Best %s model: %s (error: %.4f%%)\n', modelCategory, bestParams.modelType, bestError);
else
  fprintf('No valid model found for %s category\n', modelCategory);
end
end

function [bestParams, bestError] = optimizeSingleModelWithGA(modelType, alphaList, Ea_pred, beta, T_start_K, T_end_K, T_exp, w_exp, w_inf, gaMode)
% Use GA to optimize single model parameters
% gaMode: 'coarse' for Stage 1, 'fine' for Stage 2, 'single_model' for original behavior

if nargin < 10
  gaMode = 'single_model';  % Default to original behavior
end

paramRanges = getParameterRangesForModel(modelType);
numParams = length(paramRanges);

if numParams == 0
  bestParams = struct();
  [bestError, bestParams] = testSingleModel(modelType, bestParams, alphaList, Ea_pred, beta, T_start_K, T_end_K, T_exp, w_exp, w_inf);
  return;
end

% Set GA parameter bounds
[lb, ub] = getParameterBounds(paramRanges);

% Show GA mode info
if strcmp(gaMode, 'coarse')
  fprintf('    [Coarse] GA: %d params, range [%.2f, %.2f]\n', numParams, min(lb), max(ub));
elseif strcmp(gaMode, 'fine')
  fprintf('    [Fine] GA: %d params, range [%.2f, %.2f]\n', numParams, min(lb), max(ub));
else
  fprintf('    GA optimization: %d parameters, range [%.2f, %.2f]\n', numParams, min(lb), max(ub));
end

% Get GA configuration based on mode
options = getGAOptions(gaMode, numParams);

% Define fitness function inline
fitnessFcn = @(X) singleModelFitnessInline(X, modelType, alphaList, Ea_pred, beta, T_start_K, T_end_K, T_exp, w_exp, w_inf);

% Run GA
[bestX, bestF] = ga(fitnessFcn, numParams, [], [], [], [], lb, ub, [], options);

% Reconstruct best parameters
bestParams = createParamsStruct(modelType, bestX);
bestParams.modelType = modelType;
bestError = bestF;

if strcmp(gaMode, 'fine')
  fprintf('    [Fine] GA completed. Best error: %.4f%%, n=%.4f\n', bestError, bestX(1));
end
end

function [lb, ub] = getParameterBounds(paramRanges)
% Get parameter bounds
numParams = length(paramRanges);
lb = zeros(1, numParams);
ub = zeros(1, numParams);

for i = 1:numParams
  lb(i) = paramRanges{i}{2};  % Minimum value
  ub(i) = paramRanges{i}{3};  % Maximum value
end
end

function allModels = getAllModelsInCategory(modelCategory)
% Define all available models in each category
switch modelCategory
  case 'diffusion'
    allModels = {'parabolic_1d', 'valensi_2d', 'jander_2d', 'ginstling_brounshtein_3d', 'jander_3d', 'anti_jander_3d', 'zhuralev_lesokin_tempelman_3d'};
  case 'nucleation'
    allModels = {'avrami_erofeev', 'prout_tomkins'};
  case 'powerlaw'
    allModels = {'mapel_power'};
  case 'reaction_order'
    allModels = {'first_order', 'nth_order'};
  case 'geometrical'
    allModels = {'contracting_cylinder', 'contracting_sphere'};
  otherwise
    error('Unknown model category: %s', modelCategory);
end
end

function needs = needsParameters(modelType)
% Check if model needs parameters
switch modelType
  case {'avrami_erofeev', 'prout_tomkins', 'mapel_power', 'nth_order', 'jander_2d', 'jander_3d'}
    needs = true;
  otherwise
    needs = false;
end
end

function paramRanges = getParameterRangesForModel(modelType)
% Get parameter ranges for specific model
paramRanges = {};

switch modelType
  case 'avrami_erofeev'
    paramRanges = {{'n', 0.5, 4.0}};
  case 'prout_tomkins'
    paramRanges = {{'n', 0.5, 4.0}};
  case 'mapel_power'
    paramRanges = {{'n', 0.5, 4.0}};
  case 'nth_order'
    paramRanges = {{'n', 0.5, 4.0}};
  case 'jander_2d'
    paramRanges = {{'n', 0.5, 4.0}};
  case 'jander_3d'
    paramRanges = {{'n', 0.5, 4.0}};
  otherwise
    % No parameters needed
end
end

function params = createParamsStruct(modelType, X)
% Create parameter structure from optimization vector
params = struct();

switch modelType
  case 'avrami_erofeev'
    if length(X) >= 1
      params.n = X(1);
    end
  case 'prout_tomkins'
    if length(X) >= 1
      params.n = X(1);
    end
  case 'mapel_power'
    if length(X) >= 1
      params.n = X(1);
    end
  case 'nth_order'
    if length(X) >= 1
      params.n = X(1);
    end
  case 'jander_2d'
    if length(X) >= 1
      params.n = X(1);
    end
  case 'jander_3d'
    if length(X) >= 1
      params.n = X(1);
    end
  otherwise
    % No parameters
end
end

function [error, params] = testSingleModel(modelType, params, alphaList, Ea_pred, beta, T_start_K, T_end_K, T_exp, w_exp, w_inf)
% Test a single model and return error
try
  % Build complete model identifier
  category = getCategoryFromModelType(modelType);
  mechId = sprintf('%s|%s', category, modelType);

  % Ensure params is a struct
  if ~isstruct(params)
    params = struct();
  end

  % For models that need parameters, provide default values if not provided
  if needsParameters(modelType) && ~isfield(params, 'n')
    params.n = 1.0;  % Default value for models that need parameters
  end

  % Generate G(α) function
  G_alpha = generateGAlpha(alphaList, mechId, params);

  % Optimize T and A
  [T_opt, A_opt, opt_error] = optimizeTandA(G_alpha, Ea_pred, beta, T_start_K, T_end_K, alphaList, T_exp, w_exp, w_inf);

  % Generate predicted TG curve
  [w_pred, T_pred] = generateTGCurveFromG(G_alpha, T_opt, A_opt, Ea_pred, beta, alphaList, w_inf, T_exp);

  % Compare with experimental data
  exp_error = calculateErrorWithKinetics(w_pred, T_pred, w_exp, T_exp, w_inf, G_alpha, T_opt, A_opt, Ea_pred, beta, alphaList);

  % Combine errors
  error = 0.2 * opt_error + 1.0 * exp_error;

  % Store results
  params.T_opt = T_opt;
  params.A_opt = A_opt;
  params.opt_error = opt_error;
  params.exp_error = exp_error;

catch ME
  fprintf('    Error testing %s: %s\n', modelType, ME.message);
  error = Inf;
  params = struct();
end
end

function category = getCategoryFromModelType(modelType)
% Get category from model type
if ismember(modelType, {'parabolic_1d', 'valensi_2d', 'jander_2d', 'ginstling_brounshtein_3d', 'jander_3d', 'anti_jander_3d', 'zhuralev_lesokin_tempelman_3d'})
  category = 'diffusion';
elseif ismember(modelType, {'avrami_erofeev', 'prout_tomkins'})
  category = 'nucleation';
elseif ismember(modelType, {'mapel_power'})
  category = 'powerlaw';
elseif ismember(modelType, {'first_order', 'nth_order'})
  category = 'reaction_order';
elseif ismember(modelType, {'contracting_cylinder', 'contracting_sphere'})
  category = 'geometrical';
else
  error('Unknown model type: %s', modelType);
end
end

function error = singleModelFitnessInline(X, modelType, alphaList, Ea_pred, beta, T_start_K, T_end_K, T_exp, w_exp, w_inf)
% Inline single model fitness function
try
  % Create parameter structure
  params = createParamsStruct(modelType, X);

  % Test model
  [error, ~] = testSingleModel(modelType, params, alphaList, Ea_pred, beta, T_start_K, T_end_K, T_exp, w_exp, w_inf);

  % Add parameter reasonableness penalty
  if length(X) >= 1
    n = X(1);
    if n < 0.5 || n > 4.0
      error = error + 1000;  % Large penalty for out-of-range parameters
    end
  end

catch ME
  % If error occurs, return infinity
  error = Inf;
  fprintf('    GA objective error for %s: %s\n', modelType, ME.message);
end
end