function G_alpha = generateGAlpha(alphaList, mechId, params)
%GENERATEGALPHA  Compute G(α) for a given mechanism identifier.
%   mechId can be either a high-level category (backwards compatibility)
%   or a fully specified string "category|modelType" that pinpoints the
%   concrete mathematical model to use.

% ---------------------------------------------------------------------
% 1. Parse mechanism identifier
% ---------------------------------------------------------------------
if contains(mechId, '|')
  parts     = split(mechId, '|');
  category  = strtrim(parts{1});   % e.g. diffusion
  modelType = strtrim(parts{2});   % e.g. jander_3d
else
  % Legacy call – treat the whole string as category and fall back to
  % default modelType inside each switch branch.
  category  = mechId;
  modelType = '';
end

% ---------------------------------------------------------------------
% 2. Dispatch to the proper model file
% ---------------------------------------------------------------------
switch category
  case 'diffusion'
    if isempty(modelType); modelType = 'parabolic_1d'; end
    G_alpha = diffusionModels(alphaList, modelType, params);

  case 'nucleation'
    if isempty(modelType); modelType = 'avrami_erofeev'; end
    G_alpha = nucleationModels(alphaList, modelType, params);

  case 'powerlaw'
    if isempty(modelType); modelType = 'mapel_power'; end
    % Ensure n exists for mapel_power
    if strcmp(modelType, 'mapel_power') && ~isfield(params,'n')
      params.n = 1;  % default exponent
    end
    G_alpha = powerLawModels(alphaList, modelType, params);

  case 'geometrical'
    if isempty(modelType); modelType = 'contracting_cylinder'; end
    G_alpha = geometricalModels(alphaList, modelType, params);

  case 'reaction_order'
    if isempty(modelType); modelType = 'first_order'; end
    G_alpha = reactionOrderModels(alphaList, modelType, params);

  case 'multi'
    % -------------------------------------------------------------
    % multi keeps the original logic, but mechIds now may be
    % "category|modelType" strings
    % -------------------------------------------------------------
    switch params.mode
      case 'series'
        G_alpha = zeros(size(alphaList));
        for k = 1:numel(params.mechanisms)
          subId     = params.mechanisms{k};
          subParams = params.mechParams{k};
          mechG     = generateGAlpha(alphaList, subId, subParams);
          G_alpha   = G_alpha + mechG;
        end
      case 'parallel'
        G_alpha = zeros(size(alphaList));
        for k = 1:numel(params.mechanisms)
          subId     = params.mechanisms{k};
          subParams = params.mechParams{k};
          mechG     = generateGAlpha(alphaList, subId, subParams);
          G_alpha   = G_alpha + params.weights(k) * mechG;
        end
      case 'hybrid'
        G_alpha = zeros(size(alphaList));
        for b = 1:numel(params.branches)
          branch = params.branches{b};
          branchG = zeros(size(alphaList));
          for m = 1:numel(branch.mechanisms)
            subId     = branch.mechanisms{m};
            subParams = branch.mechParams{m};
            mechG     = generateGAlpha(alphaList, subId, subParams);
            branchG   = branchG + mechG;
          end
          G_alpha = G_alpha + params.weights(b) * branchG;
        end
      otherwise
        error('Unknown multi mode: %s', params.mode);
    end

  otherwise
    error('Unknown mechanism category: %s', category);
end
end