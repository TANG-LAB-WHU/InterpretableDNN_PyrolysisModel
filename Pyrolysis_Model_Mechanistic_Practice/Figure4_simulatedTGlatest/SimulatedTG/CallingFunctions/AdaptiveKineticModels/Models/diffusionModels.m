function G = diffusionModels(alpha, modelType, params)
% Complete diffusion model implementation
% Inputs:
%   alpha: conversion values
%   modelType: string specifying diffusion model type
%   params: struct with model parameters
% Outputs:
%   G: G(alpha) values

switch modelType
  case 'parabolic_1d'
    G = alpha.^2;
  case 'valensi_2d'
    G = alpha + (1 - alpha).*log(1 - alpha);
  case 'jander_2d'
    n = params.n;
    G = (1 - (1 - alpha).^(1/2)).^n;
  case 'ginstling_brounshtein_3d'
    G = 1 - 2*alpha/3 - (1 - alpha).^(2/3);
  case 'jander_3d'
    n = params.n;
    G = (1 - (1 - alpha).^(1/3)).^n;
  case 'anti_jander_3d'
    G = ((1 + alpha).^(1/3) - 1).^2;
  case 'zhuralev_lesokin_tempelman_3d'
    G = ((1 - alpha).^(-1/3) - 1).^2;
  otherwise
    error('Unknown diffusion model type: %s', modelType);
end
end