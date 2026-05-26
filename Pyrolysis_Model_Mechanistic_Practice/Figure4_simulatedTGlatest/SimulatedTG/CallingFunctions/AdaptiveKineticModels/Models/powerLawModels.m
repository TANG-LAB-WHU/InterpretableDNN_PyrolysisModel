function G = powerLawModels(alpha, modelType, params)
% Power law models
% Inputs:
%   alpha: conversion values
%   modelType: string specifying power law model type
%   params: struct with model parameters
% Outputs:
%   G: G(alpha) values

switch modelType
  case 'mapel_power'
    n = params.n;
    G = alpha.^n;
  otherwise
    error('Unknown power law model type: %s', modelType);
end
end