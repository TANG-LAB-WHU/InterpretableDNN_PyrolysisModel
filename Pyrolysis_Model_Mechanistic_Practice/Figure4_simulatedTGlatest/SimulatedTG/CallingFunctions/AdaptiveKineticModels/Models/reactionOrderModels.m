function G = reactionOrderModels(alpha, modelType, params)
% Reaction order models
% Inputs:
%   alpha: conversion values
%   modelType: string specifying reaction order model type
%   params: struct with model parameters
% Outputs:
%   G: G(alpha) values

switch modelType
  case 'first_order'
    G = -log(1 - alpha);
  case 'nth_order'
    n = params.n;
    if abs(n - 1) < 1e-6
      G = -log(1 - alpha);
    else
      G = ((1 - alpha).^(1-n) - 1)/(n-1);
    end
  otherwise
    error('Unknown reaction order model type: %s', modelType);
end
end