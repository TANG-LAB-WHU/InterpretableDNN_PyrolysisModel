function [MonteCarlo_Integral,PercentError] = MC_integral(T, Ea, A, N, beta)

% In these input parameters including temperature, activation energy,
% pre-exponentioal factor, the number of sampling points in MC, and heating
% rate, the units of them are K, J/mol, 1/s, 1, K/min

R = 8.314; % gas universal constant, J/(mol*K)

% Equation to integrate over. Integral of this from a to b
f = @(T) A/beta*exp(-Ea/R./T);

% Parameters
T0 =273; % initial pyrolysis temperature
M = 1.0 * max(f(linspace(T0, T))); % upper y bound = c*max_y_value from equation

% Generate dots
for i = 1:N

    % generate random pt
    x_val = rand(1) * (T - T0) + T0;
    y_val = rand(1) * M;
    % compare random against the curve
    fx = f(x_val);
    % logic statement
    if y_val < fx % under the curve-blue
        under(i, 1) = x_val;
        under(i, 2) = y_val;
    else         % above the curve-red
        above(i, 1) = x_val;
        above(i, 2) = y_val;
    end
end

% filter out zeros
under2(:, 1) = nonzeros(under(:, 1));
under2(:, 2) = nonzeros(under(:, 2));
above2(:, 1) = nonzeros(above(:, 1));
above2(:, 2) = nonzeros(above(:, 2));

% plottting, if needing to show the Monte Carlo sampling profile, just remove the documentation signal '%' but a lot of figures will be displayed if the variable T has more inputs
% figure('name', ['The used temperature ' num2str(T) ' K']);
% plot(above2(:, 1), above2(:, 2), 'ro', 'MarkerFaceColor','r');
% hold on
% plot(under2(:, 1), under2(:, 2), 'bo', 'MarkerFaceColor','b');
% title(['The used temperature ' num2str(T) ' K']);
% legend('above', 'under');


% Integral calculation
MonteCarlo_Integral = length(under2) / N * (M * (T - T0));

% MATLAB intrinsic Integral
MATLAB_Integral = integral(f, T0, T);

% Compare both difference
PercentError  = abs(MATLAB_Integral - MonteCarlo_Integral)/MATLAB_Integral *100;

end