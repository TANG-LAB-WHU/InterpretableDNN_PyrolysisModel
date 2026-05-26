function G = geometricalModels(alpha, modelType, params)
% Geometrical contraction models
% Inputs:
%   alpha: conversion values
%   modelType: string specifying geometrical model type
%   params: struct with model parameters
% Outputs:
%   G: G(alpha) values

switch modelType
  case 'contracting_cylinder'
    G = 1 - (1 - alpha).^(1/2);
  case 'contracting_sphere'
    G = 1 - (1 - alpha).^(1/3);
  otherwise
    error('Unknown geometrical model type: %s', modelType);
end
end