function [bestParams, bestError] = multiOptimization(alphaList, Ea_pred, beta, T_start_K, T_end_K, T_exp, w_exp, w_inf)
% Multi-mechanism GA optimization - Core module
% This is a function in the Core module specifically for handling multi-mechanism optimization
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

fprintf('Starting multi-mechanism optimization from Core module...\n');
% Call the multi-model optimizer from the GA module
[bestParams, bestError] = multiModelGA(alphaList, Ea_pred, beta, T_start_K, T_end_K, T_exp, w_exp, w_inf);
fprintf('Multi-mechanism optimization completed. Best error: %.4f%%\n', bestError);
end