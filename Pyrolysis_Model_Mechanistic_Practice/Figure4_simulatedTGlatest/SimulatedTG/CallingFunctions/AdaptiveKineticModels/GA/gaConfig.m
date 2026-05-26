function options = gaConfig(optimizationType, numParams)
% gaConfig  Backwards-compatible wrapper returning GA options
% NOTE: Prefer using getGAOptions.m. This wrapper keeps compatibility
options = getGAOptions(optimizationType, numParams);
end