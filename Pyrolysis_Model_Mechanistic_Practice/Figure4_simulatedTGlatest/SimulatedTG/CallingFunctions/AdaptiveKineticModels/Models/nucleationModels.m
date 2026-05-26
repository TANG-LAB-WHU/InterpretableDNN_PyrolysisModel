function G = nucleationModels(alpha, modelType, params)
% Nucleation and growth models
% Inputs:
%   alpha: conversion values
%   modelType: string specifying nucleation model type
%   params: struct with model parameters
% Outputs:
%   G: G(alpha) values

switch modelType
  case 'avrami_erofeev'
    n = params.n;
    G = (-log(1 - alpha)).^n;
  case 'prout_tomkins'
    alpha0 = 1e-4; % Standard regularized baseline for start of Prout-Tomkins nucleation
    alpha_clamped = max(alpha0, min(1 - alpha0, alpha));
    G = log(alpha_clamped./(1 - alpha_clamped)) - log(alpha0./(1 - alpha0));
    G = max(0, G); % Ensure strictly non-negative
  otherwise
    error('Unknown nucleation model type: %s', modelType);
end
end