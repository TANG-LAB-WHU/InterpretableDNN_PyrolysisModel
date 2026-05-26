function [bestModel, allResults] = modelComparison(alphaList, Ea_pred, beta, T_start_K, T_end_K, T_exp, w_exp, w_inf)
% Enhanced model comparison with multi-mechanism support
% Now using separated fitness functions for better compatibility
% Single mechanism models use GA optimization only
%
% Mode configuration (from performanceConfig):
%   singleMode=true,  multiOnlyMode=false  -> Single only
%   singleMode=false, multiOnlyMode=true   -> Multi only
%   singleMode=false, multiOnlyMode=false  -> Both

% Load configuration
config = performanceConfig();

% Determine mode
singleMode = isfield(config, 'singleMode') && config.singleMode;
multiOnlyMode = isfield(config, 'multiOnlyMode') && config.multiOnlyMode;

% Define model categories based on mode settings
if singleMode && ~multiOnlyMode
  % Single only mode
  modelCategories = {'diffusion', 'nucleation', 'powerlaw', 'geometrical', 'reaction_order'};
  fprintf('\n=== STARTING SINGLE-MODE MODEL COMPARISON ===\n');
  fprintf('singleMode enabled: Multi-mechanism optimization disabled\n');
elseif multiOnlyMode && ~singleMode
  % Multi only mode (NEW)
  modelCategories = {'multi'};
  fprintf('\n=== STARTING MULTI-ONLY MODEL COMPARISON ===\n');
  fprintf('multiOnlyMode enabled: Single-mechanism models skipped\n');
else
  % Full mode: include both single and multi-mechanism
  modelCategories = {'diffusion', 'nucleation', 'powerlaw', 'geometrical', 'reaction_order', 'multi'};
  fprintf('\n=== STARTING FULL MODEL COMPARISON ===\n');
  fprintf('Testing both single-mechanism and multi-mechanism models\n');
end

allResults = cell(length(modelCategories), 1);

fprintf('Using separated fitness functions for better compatibility\n');
fprintf('Single mechanism models use GA optimization only (no parameter combination testing)\n');

for i = 1:length(modelCategories)
  category = modelCategories{i};
  fprintf('\nTesting %s models against experimental data...\n', category);

  if strcmp(category, 'multi')
    % Use multi-mechanism optimization
    [bestParams, bestError] = multiOptimization(alphaList, Ea_pred, beta, T_start_K, T_end_K, T_exp, w_exp, w_inf);
  else
    % Use GA optimization for single models (no parameter combination testing)
    [bestParams, bestError] = singleModelGA(category, alphaList, Ea_pred, beta, T_start_K, T_end_K, T_exp, w_exp, w_inf);
  end

  allResults{i} = struct('category', category, 'params', bestParams, 'error', bestError);
end

% Find overall best model
errors = zeros(length(modelCategories), 1);
for i = 1:length(modelCategories)
  errors(i) = allResults{i}.error;
end

[~, bestIdx] = min(errors);
bestModel = allResults{bestIdx};

fprintf('\n=== OVERALL BEST MODEL ===\n');
fprintf('Category: %s\n', bestModel.category);
if isfield(bestModel.params, 'modelType')
  fprintf('Best specific model: %s\n', bestModel.params.modelType);
end
fprintf('Error vs experimental data: %.2e\n', bestModel.error);

% Display results for all categories
fprintf('\n=== ALL CATEGORY RESULTS ===\n');
for i = 1:length(modelCategories)
  fprintf('%s: %.2e', modelCategories{i}, allResults{i}.error);
  if isfield(allResults{i}.params, 'modelType')
    fprintf(' (%s)', allResults{i}.params.modelType);
  end
  if i == bestIdx
    fprintf(' [BEST]');
  end
  fprintf('\n');
end
end