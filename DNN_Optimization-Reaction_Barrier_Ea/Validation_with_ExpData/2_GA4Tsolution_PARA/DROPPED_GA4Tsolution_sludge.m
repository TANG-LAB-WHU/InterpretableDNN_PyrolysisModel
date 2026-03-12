% clc;clear all

Ea_bpDNN = SampleOut_SewageSludge; % Predicted activation energy in J/mol
beta = 10/60; % Heating rate in K/min

alpha = [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99];
G_alpha = Conversion(alpha);

% Settings of GA algorithm
nvars = 2; % specify the number of variables in the problem
A = [];
b = [];
Aeq = [];
beq = [];
lb = [293 2.5E18];
ub = [1125 inf];
nonlcon = [];
options = optimoptions("ga");

options.ConstraintTolerance = 1E-3; % default value
options.SelectionFcn = 'selectionstochunif';
options.CreationFcn = 'gacreationuniform';
options.CrossoverFcn = 'crossoverscattered';
options.MutationFcn = 'mutationadaptfeasible';

options.HybridFcn = 'patternsearch';

options.Display = 'iter';
options.PlotFcn = {@gaplotbestf, @gaplotscores};

options.FitnessLimit = 1E-33;
options.FunctionTolerance = 1e-33;
options.PopulationSize = 100;
options.MaxGenerations = 200;

%% X (1) = T in K, X(2) = A
% When alpha = 0.1
Ea1 = Ea_bpDNN(1);
for i = 1 : size(G_alpha, 2)
    fitnessFcn1 = @(X) (Integral4Tintegral(X, Ea1, beta) - G_alpha(1, i)).^2;
    lb1 = [293 2.5E11];
    [T1(i, :), fval1(i)] = ga(fitnessFcn1, nvars, A, b, Aeq, beq, lb1, ub, nonlcon, options);
end
%% 
% When alpha = 0.2
Ea2 = Ea_bpDNN(2);
for i = 1 : size(G_alpha, 2)
    fitnessFcn2 = @(X) (Integral4Tintegral(X, Ea2, beta) - G_alpha(2, i)).^2;
    lb2 = [293 1E9];
    [T2(i, :), fval2(i)] = ga(fitnessFcn2, nvars, A, b, Aeq, beq, lb2, ub, nonlcon, options);
end
%%
% When alpah = 0.3
Ea3 = Ea_bpDNN(3);
for i = 1 : size(G_alpha, 2)
    fitnessFcn3 = @(X) (Integral4Tintegral(X, Ea3, beta) - G_alpha(3, i)).^2;
    lb3 = [293 1E17];
    [T3(i, :), fval3(i)] = ga(fitnessFcn3, nvars, A, b, Aeq, beq, lb3, ub, nonlcon, options);
end
%%
% When alpah = 0.4
Ea4 = Ea_bpDNN(4);
for i = 1 : size(G_alpha, 2)
    fitnessFcn4 = @(X) (Integral4Tintegral(X, Ea4, beta) - G_alpha(4, i)).^2;
    lb4 = [293 2.5E15];
    [T4(i, :), fval4(i)] = ga(fitnessFcn4, nvars, A, b, Aeq, beq, lb4, ub, nonlcon, options);
end
%%
% When alpah = 0.5
Ea5 = Ea_bpDNN(5);
for i = 1 : size(G_alpha, 2)
    fitnessFcn5 = @(X) (Integral4Tintegral(X, Ea5, beta) - G_alpha(5, i)).^2;
    lb5 = [293 2.5E17];
    [T5(i, :), fval5(i)] = ga(fitnessFcn5, nvars, A, b, Aeq, beq, lb5, ub, nonlcon, options);
end
%%
% When alpah = 0.6
Ea6 = Ea_bpDNN(6);
for i = 1 : size(G_alpha, 2)
    fitnessFcn6 = @(X) (Integral4Tintegral(X, Ea6, beta) - G_alpha(6, i)).^2;
    lb6 = [293 2.5E17];
    [T6(i, :), fval6(i)] = ga(fitnessFcn6, nvars, A, b, Aeq, beq, lb6, ub, nonlcon, options);
end
%%
% When alpah = 0.7
Ea7 = Ea_bpDNN(7);
for i = 1 : size(G_alpha, 2)
    fitnessFcn7 = @(X) (Integral4Tintegral(X, Ea7, beta) - G_alpha(7, i)).^2;
    lb7 = [293 2.5E17];
    [T7(i, :), fval7(i)] = ga(fitnessFcn7, nvars, A, b, Aeq, beq, lb7, ub, nonlcon, options);
end
%% 
% When alpah = 0.8
Ea8 = Ea_bpDNN(8);
for i = 1 : size(G_alpha, 2)
    fitnessFcn8 = @(X) (Integral4Tintegral(X, Ea8, beta) - G_alpha(8, i)).^2;
    lb8 = [293 2.5E40];
    [T8(i, :), fval8(i)] = ga(fitnessFcn8, nvars, A, b, Aeq, beq, lb8, ub, nonlcon, options);
end
%% 
% When alpah = 0.9
Ea9 = Ea_bpDNN(9);
for i = 1 : size(G_alpha, 2)
    fitnessFcn9 = @(X) (Integral4Tintegral(X, Ea9, beta) - G_alpha(9, i)).^2;
    lb9 = [293 2.5E40];
    [T9(i, :), fval9(i)] = ga(fitnessFcn9, nvars, A, b, Aeq, beq, lb9, ub, nonlcon, options);
end
%% 
% When alpah = 0.99
Ea10 = Ea_bpDNN(10);
for i = 1 : size(G_alpha, 2)
    fitnessFcn10 = @(X) (Integral4Tintegral(X, Ea10, beta) - G_alpha(10, i)).^2;
    lb10 = [293 2.5E40];
    [T10(i, :), fval10(i)] = ga(fitnessFcn10, nvars, A, b, Aeq, beq, lb10, ub, nonlcon, options);
end
%% 
Tsol = [T1(:, 1) T2(:, 1) T3(:, 1) T4(:, 1) T5(:, 1) T6(:, 1) T7(:, 1) T8(:, 1) T9(:, 1) T10(:, 1)]';
A = [T1(:, 2) T2(:, 2) T3(:, 2) T4(:, 2) T5(:, 2) T6(:, 2) T7(:, 2) T8(:, 2) T9(:, 2) T10(:, 2)]';
%% 
load Validation_TG_sludge.mat
for j = 1 : size(G_alpha, 2)
    subplot(13, 10, j);
    plot(alpha', Tsol(:, j)-273, 'bo-', alphaTG_exp(:, 2), alphaTG_exp(:, 1), 'r-');
    title(['G(\alpha) ' num2str(j)]);
end