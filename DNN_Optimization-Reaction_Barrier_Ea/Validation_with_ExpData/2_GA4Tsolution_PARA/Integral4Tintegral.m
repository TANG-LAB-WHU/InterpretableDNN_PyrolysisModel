function y = Integral4Tintegral(X, Ea, beta)
% This function is used to calculate the temperature-dependent integral in thermochemical 
% kinetics using improved Coats-Redfern approximation
% Inputs:
%   X: optimization variables [T, A]
%   Ea: activation energy in J/mol
%   beta: heating rate in K/min

R = 8.314;  % Universal gas constant, J/(mol·K)
T = X(1);   % Temperature in K
A = X(2);   % Pre-exponential factor in 1/min

% Enhanced approximation for the temperature integral
if Ea/(R*T) > 20  % Standard Coats-Redfern for high Ea/RT values
    y = (A/beta) * (R/Ea) * T.^2 .* (1 - 2 * R * T ./ Ea) .* exp(-Ea ./ (R .* T));
else  % Improved approximation for lower Ea/RT values
    % Using a more accurate approximation formula for the temperature integral
    p = Ea/(R*T);
    y = (A/beta) * (T^2/p) * (1 - 2/p + 6/(p^2) - 24/(p^3)) * exp(-p);

end