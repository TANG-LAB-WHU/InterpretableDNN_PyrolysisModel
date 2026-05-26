function [bestParams, bestError] = multiModelGA(alphaList, Ea_pred, beta, T_start_K, T_end_K, T_exp, w_exp, w_inf)
% Multi-model GA optimizer with Integrated Mechanism and Parameter Optimization
%
% Input parameters:
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
mechCandidates = mechanismLibrary();
twoStageEnabled = isfield(cfg, 'twoStageGA') && cfg.twoStageGA.enabled;

% Read multi-mechanism configuration
maxMechs = cfg.multi.maxMechs;                    % For series/parallel
maxBranches = cfg.multi.maxBranches;              % For hybrid
maxMechsPerBranch = cfg.multi.maxMechsPerBranch;  % For hybrid

% Test all three modes
modes = {'series', 'parallel', 'hybrid'};

fprintf('Testing multi-mechanism combinations with GA optimization...\n');

if twoStageEnabled
  topN = cfg.twoStageGA.multiTopN;

  % === STAGE 1: Coarse GA Screening for all configurations ===
  fprintf('  [Stage 1] Coarse GA screening for all configurations...\n');
  allConfigs = {};
  configIdx = 1;

  for modeIdx = 1:numel(modes)
    mode = modes{modeIdx};

    if strcmp(mode, 'hybrid')
      for numBranches = 2:maxBranches
        for numMechsPerBranch = 1:maxMechsPerBranch
          config = struct('mode', mode, 'numPrimary', numBranches, 'numSecondary', numMechsPerBranch);
          [params, error] = optimizeMultiModelWithGA(mode, numBranches, numMechsPerBranch, ...
            mechCandidates, alphaList, Ea_pred, beta, T_start_K, T_end_K, T_exp, w_exp, w_inf, 'multi_coarse');
          allConfigs{configIdx} = struct('config', config, 'params', params, 'error', error);
          fprintf('    %s(%d,%d): %.4f%%\n', mode, numBranches, numMechsPerBranch, error);
          configIdx = configIdx + 1;
        end
      end
    else
      for numMechs = 2:maxMechs
        config = struct('mode', mode, 'numPrimary', numMechs, 'numSecondary', 1);
        [params, error] = optimizeMultiModelWithGA(mode, numMechs, 1, ...
          mechCandidates, alphaList, Ea_pred, beta, T_start_K, T_end_K, T_exp, w_exp, w_inf, 'multi_coarse');
        allConfigs{configIdx} = struct('config', config, 'params', params, 'error', error);
        fprintf('    %s(%d): %.4f%%\n', mode, numMechs, error);
        configIdx = configIdx + 1;
      end
    end
  end

  % Rank and select Top-N
  errors = cellfun(@(x) x.error, allConfigs);
  [sortedErrors, sortIdx] = sort(errors);
  topN = min(topN, length(allConfigs));

  fprintf('  [Stage 2] Fine GA optimization for Top-%d configurations:\n', topN);
  for rank = 1:topN
    idx = sortIdx(rank);
    cfg = allConfigs{idx}.config;
    fprintf('    #%d: %s (coarse error: %.4f%%)\n', rank, cfg.mode, sortedErrors(rank));

    % Fine optimization
    [params, error] = optimizeMultiModelWithGA(cfg.mode, cfg.numPrimary, cfg.numSecondary, ...
      mechCandidates, alphaList, Ea_pred, beta, T_start_K, T_end_K, T_exp, w_exp, w_inf, 'multi_fine');
    allConfigs{idx}.params = params;
    allConfigs{idx}.error = error;
    fprintf('        -> fine error: %.4f%%\n', error);
  end

  % Find best after Stage 2
  errors = cellfun(@(x) x.error, allConfigs);
  [bestError, bestIdx] = min(errors);
  bestParams = allConfigs{bestIdx}.params;

else
  % === Original single-stage logic ===
  bestError = Inf;
  bestParams = [];

  for modeIdx = 1:numel(modes)
    mode = modes{modeIdx};
    fprintf('  Testing %s mode...\n', mode);

    if strcmp(mode, 'hybrid')
      for numBranches = 2:maxBranches
        for numMechsPerBranch = 1:maxMechsPerBranch
          fprintf('    Testing hybrid: %d branches, %d mechs per branch\n', numBranches, numMechsPerBranch);
          [params, error] = optimizeMultiModelWithGA(mode, numBranches, numMechsPerBranch, mechCandidates, alphaList, Ea_pred, beta, T_start_K, T_end_K, T_exp, w_exp, w_inf, 'multi_model');
          fprintf('      Completed hybrid: %d branches, %d mechs per branch - Error: %.4f%%\n', numBranches, numMechsPerBranch, error);
          if error < bestError
            bestError = error;
            bestParams = params;
            fprintf('      New best: error = %.4f%%\n', error);
          end
        end
      end
    else
      for numMechs = 2:maxMechs
        fprintf('    Testing %s: %d mechanisms\n', mode, numMechs);
        [params, error] = optimizeMultiModelWithGA(mode, numMechs, 1, mechCandidates, alphaList, Ea_pred, beta, T_start_K, T_end_K, T_exp, w_exp, w_inf, 'multi_model');
        fprintf('      Completed %s: %d mechanisms - Error: %.4f%%\n', mode, numMechs, error);
        if error < bestError
          bestError = error;
          bestParams = params;
          fprintf('      New best: error = %.4f%%\n', error);
        end
      end
    end
  end
end

fprintf('Multi-mechanism optimization completed. Best error: %.4f%%\n', bestError);
end

function [bestParams, bestError] = optimizeMultiModelWithGA(mode, numPrimary, numSecondary, mechCandidates, alphaList, Ea_pred, beta, T_start_K, T_end_K, T_exp, w_exp, w_inf, gaMode)
% Use GA to optimize multi-model combination dynamically

if nargin < 13
  gaMode = 'multi_model';
end

nvars = calculateTotalConfigs(numPrimary, mode, numSecondary);
[lb, ub] = calculateMultiModelBounds(nvars, mode, numPrimary, numSecondary, numel(mechCandidates));

% Setup linear constraints for parallel weights sum = 1
Aeq = []; beq = [];
if strcmp(mode, 'parallel') || strcmp(mode, 'hybrid')
  Aeq = [ones(1, numPrimary), zeros(1, nvars - numPrimary)];
  beq = 1;
end

% -----------------------------------------------------------------
% Dynamic calculation of integer variable indices for mechanisms (IntCon)
% -----------------------------------------------------------------
if strcmp(mode, 'hybrid')
  intcon = (numPrimary + numPrimary * numSecondary + 1) : nvars;
elseif strcmp(mode, 'parallel')
  intcon = (2 * numPrimary + 1) : nvars;
else
  intcon = (numPrimary + 2) : nvars;
end

options = getGAOptions(gaMode, nvars);

fitnessFcn = @(X) multiModelFitness(X, mechCandidates, alphaList, Ea_pred, beta, T_start_K, T_end_K, T_exp, w_exp, w_inf, mode, numPrimary, numSecondary);

% Run GA with native mixed-integer constraints (intcon is the 10th parameter)
[bestX, bestF] = ga(fitnessFcn, nvars, [], [], Aeq, beq, lb, ub, [], intcon, options);

% Reconstruct parameters
bestParams = reconstructMultiModelParams(bestX, mechCandidates, mode, numPrimary, numSecondary);
bestParams.mode = mode;
bestError = bestF;
end

function total = calculateTotalConfigs(numPrimary, mode, numSecondary)
% Calculate total number of variables in GA chromosome
if strcmp(mode, 'hybrid')
  % Branch weights (numPrimary) + mechanism parameters (numPrimary * numSecondary) + mechanism selection indices (numPrimary * numSecondary)
  total = numPrimary + 2 * (numPrimary * numSecondary);
else
  if strcmp(mode, 'parallel')
    % Weights (numPrimary) + parameters (numPrimary) + mechanism selection indices (numPrimary)
    total = 3 * numPrimary;
  else
    % Weight (1) + parameters (numPrimary) + mechanism selection indices (numPrimary)
    total = 1 + 2 * numPrimary;
  end
end
end

function [lb, ub] = calculateMultiModelBounds(nvars, mode, numPrimary, numSecondary, numCandidates)
% Calculate lower and upper bounds for GA chromosome (strictly integer boundaries for IntCon)
if strcmp(mode, 'hybrid')
  lb = [zeros(1, numPrimary), 0.1 * ones(1, numPrimary * numSecondary), ones(1, numPrimary * numSecondary)];
  ub = [ones(1, numPrimary), 10 * ones(1, numPrimary * numSecondary), numCandidates * ones(1, numPrimary * numSecondary)];
else
  if strcmp(mode, 'parallel')
    lb = [zeros(1, numPrimary), 0.1 * ones(1, numPrimary), ones(1, numPrimary)];
    ub = [ones(1, numPrimary), 10 * ones(1, numPrimary), numCandidates * ones(1, numPrimary)];
  else
    lb = [0, 0.1 * ones(1, numPrimary), ones(1, numPrimary)];
    ub = [1, 10 * ones(1, numPrimary), numCandidates * ones(1, numPrimary)];
  end
end
end

function params = reconstructMultiModelParams(X, mechCandidates, mode, numPrimary, numSecondary)
% Reconstruct parameters and selected mechanisms from GA vector using safe rounding for IntCon
% Uses explicit weight normalization to guarantee strict physical mass conservation (sum = 1)
params = struct();
params.mode = mode;

if strcmp(mode, 'hybrid')
  weights = X(1:numPrimary);
  
  % Double Closed-Loop Safeguard 2 (Output End): Ensure reconstructed weights strictly satisfy physical conservation
  if sum(weights) > 0
    weights = weights / sum(weights);
  else
    weights = ones(size(weights)) / length(weights);
  end
  params.weights = weights;
  params.branches = cell(1, numPrimary);
  
  paramIdx = numPrimary + 1;
  indexIdx = numPrimary + numPrimary * numSecondary + 1;
  
  for b = 1:numPrimary
    branch = struct();
    branch.mechParams = cell(1, numSecondary);
    % Rounding is safer to shield against floating point precision noise
    mechIndices = round(X(indexIdx : indexIdx + numSecondary - 1));
    mechIndices = max(1, min(length(mechCandidates), mechIndices));
    branch.mechanisms = mechCandidates(mechIndices);
    
    for m = 1:numSecondary
      branch.mechParams{m} = struct('n', X(paramIdx));
      paramIdx = paramIdx + 1;
    end
    indexIdx = indexIdx + numSecondary;
    params.branches{b} = branch;
  end
else
  if strcmp(mode, 'parallel')
    weights = X(1:numPrimary);
    
    % Double Closed-Loop Safeguard 2 (Output End): Ensure reconstructed weights strictly satisfy physical conservation
    if sum(weights) > 0
      weights = weights / sum(weights);
    else
      weights = ones(size(weights)) / length(weights);
    end
    params.weights = weights;
    paramIdx = numPrimary + 1;
    indexIdx = 2 * numPrimary + 1;
  else
    weights = X(1);
    params.weights = weights;
    paramIdx = 2;
    indexIdx = numPrimary + 2;
  end
  
  % Rounding is safer to shield against floating point precision noise
  mechIndices = round(X(indexIdx : indexIdx + numPrimary - 1));
  mechIndices = max(1, min(length(mechCandidates), mechIndices));
  params.mechanisms = mechCandidates(mechIndices);
  params.mechParams = cell(1, numPrimary);
  
  for m = 1:numPrimary
    params.mechParams{m} = struct('n', X(paramIdx));
    paramIdx = paramIdx + 1;
  end
end
end

function error = multiModelFitness(X, mechCandidates, alphaList, Ea_pred, beta, T_start_K, T_end_K, T_exp, w_exp, w_inf, mode, numPrimary, numSecondary)
% Multi-model fitness function for GA

try
  % =========================================================================
  % Double Closed-Loop Safeguard 1 (Calculation End): Mass Conservation Safeguard
  % Force normalization of chromosome weights slice before kinetic computation
  % =========================================================================
  if strcmp(mode, 'parallel') || strcmp(mode, 'hybrid')
    weights_slice = X(1:numPrimary);
    w_sum = sum(weights_slice);
    if w_sum > 0
      X(1:numPrimary) = weights_slice / w_sum;
    else
      X(1:numPrimary) = ones(1, numPrimary) / numPrimary;
    end
  end
  % =========================================================================

  % Reconstruct parameters dynamically from the current chromosome
  params = reconstructMultiModelParams(X, mechCandidates, mode, numPrimary, numSecondary);

  % Generate G(α) function based on selected mechanisms
  G_alpha = generateGAlpha(alphaList, 'multi', params);

  % Optimize T and A (using direct physical mapping)
  [T_opt, A_opt, ~] = optimizeTandA(G_alpha, Ea_pred, beta, T_start_K, T_end_K, alphaList, T_exp, w_exp, w_inf);

  % Generate TG curve using proper kinetic integration
  [w_pred, T_pred] = generateTGCurveFromG(G_alpha, T_opt, A_opt, Ea_pred, beta, alphaList, w_inf, T_exp);

  % Calculate error comparing simulated and experimental TG curves
  error = calculateErrorWithKinetics(w_pred, T_pred, w_exp, T_exp, w_inf, G_alpha, T_opt, A_opt, Ea_pred, beta, alphaList);

  % Add parameter penalty
  error = addParameterPenalty(error, X, mode, numPrimary);

catch ME
  error = Inf;
  fprintf('    Multi-model GA objective error: %s\n', ME.message);
end
end

function error = addParameterPenalty(error, X, mode, numPrimary)
% Add parameter reasonableness penalty
penalty = 0;

if strcmp(mode, 'parallel')
  weights = X(1:numPrimary);
  if any(weights < 0) || any(weights > 1)
    penalty = penalty + 1000;
  end
  if abs(sum(weights) - 1) > 0.05
    penalty = penalty + 500;
  end
elseif strcmp(mode, 'hybrid')
  weights = X(1:numPrimary);
  if any(weights < 0) || any(weights > 1)
    penalty = penalty + 1000;
  end
  if abs(sum(weights) - 1) > 0.05
    penalty = penalty + 500;
  end
end

error = error + penalty;
end