function [isValid, validationResults] = modelValidation(modelType, params, alphaList, Ea_pred, beta, T_start_K, T_end_K, T_exp, w_exp, w_inf)
% Model Validation - Evaluation Module
% Validates model parameter physical reasonableness and prediction effectiveness
%
% Input parameters:
%   modelType - Model type
%   params - Model parameters
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
%   isValid - Whether validation passes
%   validationResults - Detailed validation results

validationResults = struct();
isValid = true;

fprintf('Validating model: %s\n', modelType);

% 1. Physical parameter reasonableness validation
[paramValid, paramResults] = validatePhysicalParameters(modelType, params);
validationResults.physicalParameters = paramResults;

if ~paramValid
  isValid = false;
  fprintf('    Physical parameter validation failed\n');
end

% 2. Model prediction capability validation
try
  % Test model prediction
  [error, ~] = testSingleModel(modelType, params, alphaList, Ea_pred, beta, T_start_K, T_end_K, T_exp, w_exp, w_inf);

  if isinf(error) || isnan(error)
    isValid = false;
    validationResults.predictionError = Inf;
    validationResults.predictionValid = false;
    fprintf('    Model prediction validation failed: infinite or NaN error\n');
  else
    validationResults.predictionError = error;
    validationResults.predictionValid = true;
    fprintf('    Model prediction validation passed: error = %.4f%%\n', error);
  end

catch ME
  isValid = false;
  validationResults.predictionError = Inf;
  validationResults.predictionValid = false;
  fprintf('    Model prediction validation failed: %s\n', ME.message);
end

% 3. Numerical stability validation
try
  % Test G(α) generation
  category = getCategoryFromModelType(modelType);
  mechId = sprintf('%s|%s', category, modelType);
  G_alpha = generateGAlpha(alphaList, mechId, params);

  % Check G(α) numerical stability
  if any(isnan(G_alpha)) || any(isinf(G_alpha))
    isValid = false;
    validationResults.numericalStability = false;
    fprintf('    Numerical stability validation failed: NaN or Inf in G(α)\n');
  else
    validationResults.numericalStability = true;
    validationResults.gAlphaRange = [min(G_alpha), max(G_alpha)];
    fprintf('    Numerical stability validation passed: G(α) range [%.4f, %.4f]\n', min(G_alpha), max(G_alpha));
  end

catch ME
  isValid = false;
  validationResults.numericalStability = false;
  fprintf('    Numerical stability validation failed: %s\n', ME.message);
end

% Summary
if isValid
  fprintf('✓ Model validation passed\n');
else
  fprintf('✗ Model validation failed\n');
end
end

function [isValid, results] = validatePhysicalParameters(modelType, params)
% Validate physical parameter reasonableness
% Input: modelType, params
% Output: isValid, results

isValid = true;
results = struct();

switch modelType
  case 'avrami_erofeev'
    if isfield(params, 'n')
      if params.n < 0.5 || params.n > 4
        isValid = false;
        results.n = 'Reaction order n should be between 0.5 and 4';
      end
    end

  case 'prout_tomkins'
    if isfield(params, 'n')
      if params.n < 0.5 || params.n > 4
        isValid = false;
        results.n = 'Reaction order n should be between 0.5 and 4';
      end
    end

  case 'mapel_power'
    if isfield(params, 'n')
      if params.n < 0.5 || params.n > 4
        isValid = false;
        results.n = 'Reaction order n should be between 0.5 and 4';
      end
    end

  case 'nth_order'
    if isfield(params, 'n')
      if params.n < 0.5 || params.n > 4
        isValid = false;
        results.n = 'Reaction order n should be between 0.5 and 4';
      end
    end

  otherwise
    % No parameter models
    results.message = 'No parameters to validate';
end
end

function category = getCategoryFromModelType(modelType)
% Get category from model type
% Input: modelType
% Output: category

if contains(modelType, 'diffusion') || contains(modelType, 'parabolic') || contains(modelType, 'valensi') || ...
    contains(modelType, 'jander') || contains(modelType, 'ginstling') || contains(modelType, 'zhuralev')
  category = 'diffusion';
elseif contains(modelType, 'nucleation') || contains(modelType, 'avrami') || contains(modelType, 'prout')
  category = 'nucleation';
elseif contains(modelType, 'power') || contains(modelType, 'mapel')
  category = 'powerlaw';
elseif contains(modelType, 'order') || contains(modelType, 'first')
  category = 'reaction_order';
elseif contains(modelType, 'geometrical') || contains(modelType, 'contracting')
  category = 'geometrical';
else
  error('Unknown model type: %s', modelType);
end
end