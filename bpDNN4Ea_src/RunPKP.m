%% Pyrolysis model (PyroM) for energy requirement and char yield prediction
clc, clear;
time_step = [];% preallocate storage space for speed and record time elapsed at each step of the below codes
%% Step 1 Build in relationship of activation enetgy Ea and pre-exponential factor A with sewage sludge composition information collected through literature review
% Here DEEP NEURAL NETWORK with Error BACKPROPAGATION (bp-DNN) was adopted to complete the establishment. 
tic;
run bpDNN4Ea.m   
X_new = [29.75, 3.61, 10.94, 4.25, 0.96, 40.99, 8.52, 50.49]; % input the sludge sample to be simulated
Ea_new  = nnpredict(net, X_new')*1000

% With the complexity of sludge components and lumped reactions, compensation effect between lnA and Ea generally exist. Here lnA=0.1476Ea + 1.7012 was adopted based on literature data
A_f = @(x) exp(0.1476*x/1000 + 1.7012); % in 1/s
A_new = feval(A_f, Ea_new) % determine A based on the above Ea
time_step(:,1) = toc;
fprintf(['The time elapsed in Step 1 was ' num2str(time_step(:,1)) ' seconds.\n']) % show time elapsed in this step
%% Step 2 Monte Carlo sampling method for calculating the temperature-dependent integral 
tic;
beta = 10/60;% heating rate in K/s, engaging in SENSITIVITY ANALYSIS
N = 50000;% the number of Monte Carlo sampling points for integrating temperature-dependent item. Higher N can gain more accurate result but elapse more time, so tradeoff between accurancy and acceptable running time should be considered for N. 
Y = [];% preallocate storage space for speed
Y_percent = [];% preallocate storage space for speed
% T = ValidationData(:,1);
T = 298:5:1473;% pyrolysis temperature domain interested. For example, '298', '10', and '1273' denotes the starting temperature (K), incremental step (K), and final temperature (K), respectively
T_STEP = length(T);% retrieve the amount of discrete T values for following syntax
for i = 1 : T_STEP
    [Y(i), Y_percent(i)] = MC_integral(T(i), Ea_new, A_new, N, beta);% MC_integral is directly called in the form of .m function. You can open it at the current work folder to see detailed code
end
figure % creating a figure window to show the calculated temperature-dependent integral and relative error compared to MATLAB intrnsic integral algorithm
yyaxis left
plot(T-273, Y, 'bo');xlabel('Temperature/^oC', 'FontSize', 14); ylabel('The temperature-dependent integral', 'FontSize', 14);
yyaxis right
plot(T-273, Y_percent, '--r^');ylabel('Relative error with regard to MATLAB integral/%', 'FontSize', 14)
title('Monte Carlo approach for temperature-dependent integral item');
time_step(:, 2) = toc;
fprintf(['The time elapsed in Step 2 was ' num2str(time_step(:, 2)) ' seconds.\n']) % show time elapsed in this step
%% Step 3 Align alpha with pyrolysis temperature
tic; 
syms alpha
assume(0 <= alpha <= 1) % refine the calculated value of the degree of conversion
N_Y = length(Y); % retrieve the number of the determined temperature-dependent integral
sol_alpha = []; % preallocate storage space for speed

% Preassign empty matrix to save the output of calculated values of alpha as described below
s_alpha1 = zeros(1, N_Y); s_alpha2 = zeros(1, N_Y); s_alpha3 = zeros(1, N_Y); s_alpha4 = zeros(1, N_Y); s_alpha5 = zeros(1, N_Y);
s_alpha6 = zeros(1, N_Y); s_alpha7 = zeros(1, N_Y); s_alpha8 = zeros(1, N_Y); s_alpha9 = zeros(1, N_Y); s_alpha10 = zeros(1, N_Y);
s_alpha11 = zeros(1, N_Y); s_alpha12 = zeros(1, N_Y); s_alpha13 = zeros(1, N_Y); s_alpha14 = zeros(1, N_Y); s_alpha15 = zeros(1, N_Y);
s_alpha16 = zeros(1, N_Y); s_alpha17 = zeros(1, N_Y); s_alpha18 = zeros(1, N_Y); s_alpha19 = zeros(1, N_Y); s_alpha20 = zeros(1, N_Y);
s_alpha21 = zeros(1, N_Y); s_alpha22 = zeros(1, N_Y); s_alpha23 = zeros(1, N_Y); s_alpha24 = zeros(1, N_Y); s_alpha25 = zeros(1, N_Y);
s_alpha26 = zeros(1, N_Y); s_alpha27 = zeros(1, N_Y); s_alpha28 = zeros(1, N_Y); s_alpha29 = zeros(1, N_Y); s_alpha30 = zeros(1, N_Y);
s_alpha31 = zeros(1, N_Y); s_alpha32 = zeros(1, N_Y); s_alpha33 = zeros(1, N_Y); s_alpha34 = zeros(1, N_Y); s_alpha35 = zeros(1, N_Y);
s_alpha36 = zeros(1, N_Y); s_alpha37 = zeros(1, N_Y); s_alpha38 = zeros(1, N_Y); s_alpha39 = zeros(1, N_Y); s_alpha40 = zeros(1, N_Y);
s_alpha41 = zeros(1, N_Y); s_alpha42 = zeros(1, N_Y); s_alpha43 = zeros(1, N_Y); s_alpha44 = zeros(1, N_Y); s_alpha45 = zeros(1, N_Y);
s_alpha46 = zeros(1, N_Y); s_alpha47 = zeros(1, N_Y); s_alpha48 = zeros(1, N_Y); s_alpha49 = zeros(1, N_Y); s_alpha50 = zeros(1, N_Y);
s_alpha51 = zeros(1, N_Y); s_alpha52 = zeros(1, N_Y); s_alpha53 = zeros(1, N_Y); s_alpha54 = zeros(1, N_Y); s_alpha55 = zeros(1, N_Y);
s_alpha56 = zeros(1, N_Y); s_alpha57 = zeros(1, N_Y); s_alpha58 = zeros(1, N_Y); s_alpha59 = zeros(1, N_Y); s_alpha60 = zeros(1, N_Y);
s_alpha61 = zeros(1, N_Y); s_alpha62 = zeros(1, N_Y); s_alpha63 = zeros(1, N_Y); s_alpha64 = zeros(1, N_Y); s_alpha65 = zeros(1, N_Y);
s_alpha66 = zeros(1, N_Y); s_alpha67 = zeros(1, N_Y); s_alpha68 = zeros(1, N_Y); s_alpha69 = zeros(1, N_Y); s_alpha70 = zeros(1, N_Y);
s_alpha71 = zeros(1, N_Y); s_alpha72 = zeros(1, N_Y); s_alpha73 = zeros(1, N_Y); s_alpha74 = zeros(1, N_Y); s_alpha75 = zeros(1, N_Y);
s_alpha76 = zeros(1, N_Y); s_alpha77 = zeros(1, N_Y); s_alpha78 = zeros(1, N_Y); s_alpha79 = zeros(1, N_Y); s_alpha80 = zeros(1, N_Y);
s_alpha81 = zeros(1, N_Y); s_alpha82 = zeros(1, N_Y); s_alpha83 = zeros(1, N_Y); s_alpha84 = zeros(1, N_Y); s_alpha85 = zeros(1, N_Y);
s_alpha86 = zeros(1, N_Y); s_alpha87 = zeros(1, N_Y); s_alpha88 = zeros(1, N_Y); s_alpha89 = zeros(1, N_Y); s_alpha90 = zeros(1, N_Y);
s_alpha91 = zeros(1, N_Y); s_alpha92 = zeros(1, N_Y); s_alpha93 = zeros(1, N_Y); s_alpha94 = zeros(1, N_Y); s_alpha95 = zeros(1, N_Y);
s_alpha96 = zeros(1, N_Y); s_alpha97 = zeros(1, N_Y); s_alpha98 = zeros(1, N_Y); s_alpha99 = zeros(1, N_Y); s_alpha100 = zeros(1, N_Y);
s_alpha101 = zeros(1, N_Y); s_alpha102 = zeros(1, N_Y); s_alpha103 = zeros(1, N_Y); s_alpha104 = zeros(1, N_Y);

% Pre-associate alpha values with the temperature-dependent interal
for i = 1 : N_Y
    ss_eqn(i, :) = Conversion(alpha) == Y(i);
end

% Determine alpha values in line with the used mechanistic models
for  i = 1: N_Y
    try
       s1(i) = ss_eqn(i, 1);
       s_alpha1(i) = double(solve(s1(i), alpha));
    catch
        i;
    end
end
 
for  i = 1: N_Y
    try
       s2(i) = ss_eqn(i, 2);
       s_alpha2(i) = double(solve(s2(i), alpha));
    catch
        i;
    end
end

for  i = 1: N_Y
    try
       s3(i) = ss_eqn(i, 3);
       s_alpha3(i) = double(solve(s3(i), alpha));
    catch
        i;
    end
end

for  i = 1: N_Y
    try
       s4(i) = ss_eqn(i, 4);
       s_alpha4(i) = double(solve(s4(i), alpha));
    catch
        i;
    end
end

for  i = 1: N_Y
    try
       s5(i) = ss_eqn(i, 5);
       s_alpha5(i) = double(solve(s5(i), alpha));
    catch
        i;
    end
end

for  i = 1: N_Y
    try
       s6(i) = ss_eqn(i, 6);
       s_alpha6(i) = double(solve(s6(i), alpha));
    catch
        i;
    end
end

for  i = 1: N_Y
    try
       s7(i) = ss_eqn(i, 7);
       s_alpha7(i) = double(solve(s7(i), alpha));
    catch
        i;
    end
end

for  i = 1: N_Y
    try
       s8(i) = ss_eqn(i, 8);
       s_alpha8(i) = double(solve(s8(i), alpha));
    catch
        i;
    end
end

for  i = 1: N_Y
    try
       s9(i) = ss_eqn(i, 9);
       s_alpha9(i) = double(solve(s9(i), alpha));
    catch
        i;
    end
end

for  i = 1: N_Y
    try
       s10(i) = ss_eqn(i, 10);
       s_alpha10(i) = double(solve(s10(i), alpha));
    catch
        i;
    end
end

for  i = 1: N_Y
    try
       s11(i) = ss_eqn(i, 11);
       s_alpha11(i) = double(solve(s11(i), alpha));
    catch
        i;
    end
end

for  i = 1: N_Y
    try
       s12(i) = ss_eqn(i, 12);
       s_alpha12(i) = double(solve(s12(i), alpha));
    catch
        i;
    end
end

for  i = 1: N_Y
    try
       s13(i) = ss_eqn(i, 13);
       s_alpha13(i) = double(solve(s13(i), alpha));
    catch
        i;
    end
end
 
for  i = 1: N_Y
    try
       s14(i) = ss_eqn(i, 14);
       s_alpha14(i) = double(solve(s14(i), alpha));
    catch
        i; 
    end
end
 
for  i = 1: N_Y
    try
       s15(i) = ss_eqn(i, 15);
       s_alpha15(i) = double(solve(s15(i), alpha));
    catch
        i; 
    end
end
 
for  i = 1: N_Y
    try
       s16(i) = ss_eqn(i, 16);
       s_alpha16(i) = double(solve(s16(i), alpha));
    catch
        i; 
    end
end
 
for  i = 1: N_Y
    try
       s17(i) = ss_eqn(i, 17);
       s_alpha17(i) = double(solve(s17(i), alpha));
    catch
        i; 
    end
end

for  i = 1: N_Y
    try
       s18(i) = ss_eqn(i, 18);
       s_alpha18(i) = double(solve(s18(i), alpha));
    catch
        i; 
    end
end
 
for  i = 1: N_Y
    try
       s19(i) = ss_eqn(i, 19);
       s_alpha19(i) = double(solve(s19(i), alpha));
    catch
        i; 
    end
end
 
for  i = 1: N_Y
    try
       s20(i) = ss_eqn(i, 20);
       s_alpha20(i) = double(solve(s20(i), alpha));
    catch
        i; 
    end
end
 
for  i = 1: N_Y
    try
       s21(i) = ss_eqn(i, 21);
       s_alpha21(i) = double(solve(s21(i), alpha));
    catch
        i;
    end
end
 
for  i = 1: N_Y
    try
       s22(i) = ss_eqn(i, 22);
       s_alpha22(i) = double(solve(s22(i), alpha));
    catch
        i;
    end
end
 
for  i = 1: N_Y
    try
       s23(i) = ss_eqn(i, 23);
       s_alpha23(i) = double(solve(s23(i), alpha));
    catch
        i;
    end
end
 
for  i = 1: N_Y
    try
       s24(i) = ss_eqn(i, 24);
       s_alpha24(i) = double(solve(s24(i), alpha));
    catch
        i; 
    end
end
 
for  i = 1: N_Y
    try
       s25(i) = ss_eqn(i, 25);
       s_alpha25(i) = double(solve(s25(i), alpha));
    catch
        i; 
    end
end
 
for  i = 1: N_Y
    try
       s26(i) = ss_eqn(i, 26);
       s_alpha26(i) = double(solve(s26(i), alpha));
    catch
        i; 
    end
end

for  i = 1: N_Y
    try
       s27(i) = ss_eqn(i, 27);
       s_alpha27(i) = double(solve(s27(i), alpha));
    catch
        i; 
    end
end
 
for  i = 1: N_Y
    try
       s28(i) = ss_eqn(i, 28);
       s_alpha28(i) = double(solve(s28(i), alpha));
    catch
        i; 
    end
end
 
for  i = 1: N_Y
    try
       s29(i) = ss_eqn(i, 29);
       s_alpha29(i) = double(solve(s29(i), alpha));
    catch
        i; 
    end
end
 
for  i = 1: N_Y
    try
       s30(i) = ss_eqn(i, 30);
       s_alpha30(i) = double(solve(s30(i), alpha));
    catch
        i; 
    end
end
 
for  i = 1: N_Y
    try
       s31(i) = ss_eqn(i, 31);
       s_alpha31(i) = double(solve(s31(i), alpha));
    catch
        i;
    end
end

for  i = 1: N_Y
    try
       s32(i) = ss_eqn(i, 32);
       s_alpha22(i) = double(solve(s22(i), alpha));
    catch
        i;
    end
end
 
for  i = 1: N_Y
    try
       s33(i) = ss_eqn(i, 33);
       s_alpha33(i) = double(solve(s33(i), alpha));
    catch
        i;
    end
end
 
for  i = 1: N_Y
    try
       s34(i) = ss_eqn(i, 34);
       s_alpha34(i) = double(solve(s34(i), alpha));
    catch
        i; 
    end
end

for  i = 1: N_Y
    try
       s35(i) = ss_eqn(i, 35);
       s_alpha35(i) = double(solve(s35(i), alpha));
    catch
        i; 
    end
end

for  i = 1: N_Y
    try
       s36(i) = ss_eqn(i, 36);
       s_alpha36(i) = double(solve(s36(i), alpha));
    catch
        i; 
    end
end
 
for  i = 1: N_Y
    try
       s37(i) = ss_eqn(i, 37);
       s_alpha37(i) = double(solve(s37(i), alpha));
    catch
        i; 
    end
end

for  i = 1: N_Y
    try
       s38(i) = ss_eqn(i, 38);
       s_alpha38(i) = double(solve(s38(i), alpha));
    catch
        i; 
    end
end

for  i = 1: N_Y
    try
       s39(i) = ss_eqn(i, 39);
       s_alpha39(i) = double(solve(s39(i), alpha));
    catch
        i; 
    end
end
 
for  i = 1: N_Y
    try
       s40(i) = ss_eqn(i, 40);
       s_alpha40(i) = double(solve(s40(i), alpha));
    catch
        i; 
    end
end

for  i = 1: N_Y
    try
       s41(i) = ss_eqn(i, 41);
       s_alpha41(i) = double(solve(s41(i), alpha));
    catch
        i;
    end
end
 
for  i = 1: N_Y
    try
       s42(i) = ss_eqn(i, 42);
       s_alpha42(i) = double(solve(s42(i), alpha));
    catch
        i;
    end
end

for  i = 1: N_Y
    try
       s43(i) = ss_eqn(i, 43);
       s_alpha43(i) = double(solve(s43(i), alpha));
    catch
        i;
    end
end

for  i = 1: N_Y
    try
       s44(i) = ss_eqn(i, 44);
       s_alpha44(i) = double(solve(s44(i), alpha));
    catch
        i;
    end
end

for  i = 1: N_Y
    try
       s45(i) = ss_eqn(i, 45);
       s_alpha45(i) = double(solve(s45(i), alpha));
    catch
        i;
    end
end

for  i = 1: N_Y
    try
       s46(i) = ss_eqn(i, 46);
       s_alpha46(i) = double(solve(s46(i), alpha));
    catch
        i;
    end
end

for  i = 1: N_Y
    try
       s47(i) = ss_eqn(i, 47);
       s_alpha47(i) = double(solve(s47(i), alpha));
    catch
        i;
    end
end

for  i = 1: N_Y
    try
       s48(i) = ss_eqn(i, 48);
       s_alpha48(i) = double(solve(s48(i), alpha));
    catch
        i;
    end
end

for  i = 1: N_Y
    try
       s49(i) = ss_eqn(i, 49);
       s_alpha49(i) = double(solve(s49(i), alpha));
    catch
        i;
    end
end

for  i = 1: N_Y
    try
       s50(i) = ss_eqn(i, 50);
       s_alpha50(i) = double(solve(s50(i), alpha));
    catch
        i; 
    end
end

for  i = 1: N_Y
    try
       s51(i) = ss_eqn(i, 51);
       s_alpha51(i) = double(solve(s51(i), alpha));
    catch
        i;
    end
end
 
for  i = 1: N_Y
    try
       s52(i) = ss_eqn(i, 52);
       s_alpha52(i) = double(solve(s52(i), alpha));
    catch
        i;
    end
end

for  i = 1: N_Y
    try
       s53(i) = ss_eqn(i, 53);
       s_alpha53(i) = double(solve(s53(i), alpha));
    catch
        i;
    end
end

for  i = 1: N_Y
    try
       s54(i) = ss_eqn(i, 54);
       s_alpha54(i) = double(solve(s54(i), alpha));
    catch
        i;
    end
end

for  i = 1: N_Y
    try
       s55(i) = ss_eqn(i, 55);
       s_alpha55(i) = double(solve(s55(i), alpha));
    catch
        i;
    end
end

for  i = 1: N_Y
    try
       s56(i) = ss_eqn(i, 56);
       s_alpha56(i) = double(solve(s56(i), alpha));
    catch
        i;
    end
end

for  i = 1: N_Y
    try
       s57(i) = ss_eqn(i, 57);
       s_alpha57(i) = double(solve(s57(i), alpha));
    catch
        i;
    end
end

for  i = 1: N_Y
    try
       s58(i) = ss_eqn(i, 58);
       s_alpha58(i) = double(solve(s58(i), alpha));
    catch
        i;
    end
end

for  i = 1: N_Y
    try
       s59(i) = ss_eqn(i, 59);
       s_alpha59(i) = double(solve(s59(i), alpha));
    catch
        i;
    end
end

for  i = 1: N_Y
    try
       s60(i) = ss_eqn(i, 60);
       s_alpha60(i) = double(solve(s60(i), alpha));
    catch
        i; 
    end
end

for  i = 1: N_Y
    try
       s61(i) = ss_eqn(i, 61);
       s_alpha61(i) = double(solve(s61(i), alpha));
    catch
        i;
    end
end
 
for  i = 1: N_Y
    try
       s62(i) = ss_eqn(i, 62);
       s_alpha62(i) = double(solve(s62(i), alpha));
    catch
        i;
    end
end

for  i = 1: N_Y
    try
       s63(i) = ss_eqn(i, 63);
       s_alpha63(i) = double(solve(s63(i), alpha));
    catch
        i;
    end
end

for  i = 1: N_Y
    try
       s64(i) = ss_eqn(i, 64);
       s_alpha64(i) = double(solve(s64(i), alpha));
    catch
        i;
    end
end

for  i = 1: N_Y
    try
       s65(i) = ss_eqn(i, 65);
       s_alpha65(i) = double(solve(s65(i), alpha));
    catch
        i;
    end
end

for  i = 1: N_Y
    try
       s66(i) = ss_eqn(i, 66);
       s_alpha66(i) = double(solve(s66(i), alpha));
    catch
        i;
    end
end

for  i = 1: N_Y
    try
       s67(i) = ss_eqn(i, 67);
       s_alpha67(i) = double(solve(s67(i), alpha));
    catch
        i;
    end
end

for  i = 1: N_Y
    try
       s68(i) = ss_eqn(i, 68);
       s_alpha68(i) = double(solve(s68(i), alpha));
    catch
        i;
    end
end

for  i = 1: N_Y
    try
       s69(i) = ss_eqn(i, 69);
       s_alpha69(i) = double(solve(s69(i), alpha));
    catch
        i;
    end
end

for  i = 1: N_Y
    try
       s70(i) = ss_eqn(i, 70);
       s_alpha70(i) = double(solve(s70(i), alpha));
    catch
        i; 
    end
end

for  i = 1: N_Y
    try
       s71(i) = ss_eqn(i, 71);
       s_alpha71(i) = double(solve(s71(i), alpha));
    catch
        i;
    end
end
 
for  i = 1: N_Y
    try
       s72(i) = ss_eqn(i, 72);
       s_alpha72(i) = double(solve(s72(i), alpha));
    catch
        i;
    end
end

for  i = 1: N_Y
    try
       s73(i) = ss_eqn(i, 73);
       s_alpha73(i) = double(solve(s73(i), alpha));
    catch
        i;
    end
end

for  i = 1: N_Y
    try
       s74(i) = ss_eqn(i, 74);
       s_alpha74(i) = double(solve(s74(i), alpha));
    catch
        i;
    end
end

for  i = 1: N_Y
    try
       s75(i) = ss_eqn(i, 75);
       s_alpha75(i) = double(solve(s75(i), alpha));
    catch
        i;
    end
end

for  i = 1: N_Y
    try
       s76(i) = ss_eqn(i, 76);
       s_alpha76(i) = double(solve(s76(i), alpha));
    catch
        i;
    end
end

for  i = 1: N_Y
    try
       s77(i) = ss_eqn(i, 77);
       s_alpha77(i) = double(solve(s77(i), alpha));
    catch
        i;
    end
end

for  i = 1: N_Y
    try
       s78(i) = ss_eqn(i, 78);
       s_alpha78(i) = double(solve(s78(i), alpha));
    catch
        i;
    end
end

for  i = 1: N_Y
    try
       s79(i) = ss_eqn(i, 79);
       s_alpha79(i) = double(solve(s79(i), alpha));
    catch
        i;
    end
end

for  i = 1: N_Y
    try
       s80(i) = ss_eqn(i, 80);
       s_alpha80(i) = double(solve(s80(i), alpha));
    catch
        i; 
    end
end

for  i = 1: N_Y
    try
       s81(i) = ss_eqn(i, 81);
       s_alpha81(i) = double(solve(s81(i), alpha));
    catch
        i;
    end
end
 
for  i = 1: N_Y
    try
       s82(i) = ss_eqn(i, 82);
       s_alpha82(i) = double(solve(s82(i), alpha));
    catch
        i;
    end
end

for  i = 1: N_Y
    try
       s83(i) = ss_eqn(i, 83);
       s_alpha83(i) = double(solve(s83(i), alpha));
    catch
        i;
    end
end

for  i = 1: N_Y
    try
       s84(i) = ss_eqn(i, 84);
       s_alpha84(i) = double(solve(s84(i), alpha));
    catch
        i;
    end
end

for  i = 1: N_Y
    try
       s85(i) = ss_eqn(i, 85);
       s_alpha85(i) = double(solve(s85(i), alpha));
    catch
        i;
    end
end

for  i = 1: N_Y
    try
       s86(i) = ss_eqn(i, 86);
       s_alpha86(i) = double(solve(s86(i), alpha));
    catch
        i;
    end
end

for  i = 1: N_Y
    try
       s87(i) = ss_eqn(i, 87);
       s_alpha87(i) = double(solve(s87(i), alpha));
    catch
        i;
    end
end

for  i = 1: N_Y
    try
       s88(i) = ss_eqn(i, 88);
       s_alpha88(i) = double(solve(s88(i), alpha));
    catch
        i;
    end
end

for  i = 1: N_Y
    try
       s89(i) = ss_eqn(i, 89);
       s_alpha89(i) = double(solve(s89(i), alpha));
    catch
        i;
    end
end

for  i = 1: N_Y
    try
       s90(i) = ss_eqn(i, 90);
       s_alpha90(i) = double(solve(s90(i), alpha));
    catch
        i; 
    end
end

for  i = 1: N_Y
    try
       s91(i) = ss_eqn(i, 91);
       s_alpha91(i) = double(solve(s91(i), alpha));
    catch
        i;
    end
end
 
for  i = 1: N_Y
    try
       s92(i) = ss_eqn(i, 92);
       s_alpha92(i) = double(solve(s92(i), alpha));
    catch
        i;
    end
end

for  i = 1: N_Y
    try
       s93(i) = ss_eqn(i, 93);
       s_alpha93(i) = double(solve(s93(i), alpha));
    catch
        i;
    end
end

for  i = 1: N_Y
    try
       s94(i) = ss_eqn(i, 94);
       s_alpha94(i) = double(solve(s94(i), alpha));
    catch
        i;
    end
end

for  i = 1: N_Y
    try
       s95(i) = ss_eqn(i, 95);
       s_alpha95(i) = double(solve(s95(i), alpha));
    catch
        i;
    end
end

for  i = 1: N_Y
    try
       s96(i) = ss_eqn(i, 96);
       s_alpha96(i) = double(solve(s96(i), alpha));
    catch
        i;
    end
end

for  i = 1: N_Y
    try
       s97(i) = ss_eqn(i, 97);
       s_alpha97(i) = double(solve(s97(i), alpha));
    catch
        i;
    end
end

for  i = 1: N_Y
    try
       s98(i) = ss_eqn(i, 98);
       s_alpha98(i) = double(solve(s98(i), alpha));
    catch
        i;
    end
end

for  i = 1: N_Y
    try
       s99(i) = ss_eqn(i, 99);
       s_alpha99(i) = double(solve(s99(i), alpha));
    catch
        i;
    end
end

for  i = 1: N_Y
    try
       s100(i) = ss_eqn(i, 100);
       s_alpha100(i) = double(solve(s100(i), alpha));
    catch
        i; 
    end
end

for  i = 1: N_Y
    try
       s101(i) = ss_eqn(i, 101);
       s_alpha101(i) = double(solve(s101(i), alpha));
    catch
        i; 
    end
end

for  i = 1: N_Y
    try
       s102(i) = ss_eqn(i, 102);
       s_alpha102(i) = double(solve(s102(i), alpha));
    catch
        i; 
    end
end

for  i = 1: N_Y
    try
       s103(i) = ss_eqn(i, 103);
       s_alpha103(i) = double(solve(s103(i), alpha));
    catch
        i; 
    end
end

for  i = 1: N_Y
    try
       s104(i) = ss_eqn(i, 104);
       s_alpha104(i) = double(solve(s104(i), alpha));
    catch
        i; 
    end
end

% Collect the calculated alpha values in line with these kinetic models specified
sol_alpha =real([s_alpha1' s_alpha2' s_alpha3' s_alpha4' s_alpha5' s_alpha6' s_alpha7' s_alpha8'...
    s_alpha9' s_alpha10' s_alpha11' s_alpha12' s_alpha13' s_alpha14' s_alpha15' s_alpha16'...
    s_alpha17' s_alpha18' s_alpha19' s_alpha20' s_alpha21' s_alpha22' s_alpha23' s_alpha24'...
    s_alpha25' s_alpha26' s_alpha27' s_alpha28' s_alpha29' s_alpha30' s_alpha31' s_alpha32'...
    s_alpha33' s_alpha34' s_alpha35' s_alpha36' s_alpha37' s_alpha38' s_alpha39' s_alpha40'...
    s_alpha41' s_alpha42' s_alpha43' s_alpha44' s_alpha45' s_alpha46' s_alpha47' s_alpha48'...
    s_alpha49' s_alpha50' s_alpha51' s_alpha52' s_alpha53' s_alpha54' s_alpha55' s_alpha56'...
    s_alpha57' s_alpha58' s_alpha59' s_alpha60' s_alpha61' s_alpha62' s_alpha63' s_alpha64'...
    s_alpha65' s_alpha66' s_alpha67' s_alpha68' s_alpha69' s_alpha70' s_alpha71' s_alpha72'...
    s_alpha73' s_alpha74' s_alpha75' s_alpha76' s_alpha77' s_alpha78' s_alpha79' s_alpha80'...
    s_alpha81' s_alpha82' s_alpha83' s_alpha84' s_alpha85' s_alpha86' s_alpha87' s_alpha88'...
    s_alpha89' s_alpha90' s_alpha91' s_alpha92' s_alpha93' s_alpha94' s_alpha95' s_alpha96'...
    s_alpha97' s_alpha98' s_alpha99' s_alpha100' s_alpha101' s_alpha102' s_alpha103' s_alpha104']);

figure % Wholly seeing kinetic model fitting performance
plot(T-273, sol_alpha); title(['Simulated TG curve with ' num2str(length(sol_alpha(end, :))) ' kinetic models']); 
xlabel('Pyrolysis temperature/^oC'); ylabel('The degree of conversion \alpha');

figure('Name', 'Separate alpha simulated by selected kinetic models') % for separately overseeing kinetics-fitting performance
for i = 1 : length(sol_alpha(end, :))
    subplot(13, 8, i); 
    plot(T-273, sol_alpha(:,i));
    title(['G(\alpha) ' num2str(i)]);
end
time_step(:, 3) = toc;
fprintf(['The time elapsed in Step 3 was ' num2str(time_step(:, 3)) ' seconds.\n']) % show time elapsed in this step
%%  Step FOUR Calculate weight w in accordance with w = 1 - (1 - w_inf)*alpha
tic;
w_inf = X_new(end)/100; % here assume the final yiled of char to be the content of ash in sludge sample at ~ 1000 Celsius degree, which coincides with those expermental TG curves in literature
% w_inf = ValidationData(end, end)/100; % retrieve the yield production from validation dataset at the end of the teperature domain
w_T = normalizedweight(w_inf, sol_alpha);
figure % for wholly seeing simulated TG curves
plot(T-273, w_T);
figure
for i = 1 : length(w_T(end, :))
    subplot(13, 8, i); 
    plot(T-273, w_T(:,i)); 
    title(['G(\alpha) ' num2str(i)]);
end

% plot(T-273, w_T, ValidationData(:,1)-273, ValidationData(:,2),'ro','MarkerSize', 8);
% title(['Simulated TG curve with ' num2str(length(w_T)) ' kinetic models']); xlabel('Pyrolysis temperature/^oC'); ylabel('Normalized weight/%');
% figure('Name', 'Separate TG curves simulated') % for separately overseeing simulated TG curves
% % Calculate the root mean square error (RMSE) between the calculate       d values and experimental ones from literature for validation
% RMSE = []; % pre-allocate storage space for speed
% for i = 1 : length(w_T)
%     RMSE(:,i) = sqrt(sum((w_T(:,i) - ValidationData(:,2)).^2)./length(w_T));
% end
% OptimalFunctionIndice = find(RMSE == min(RMSE)) % find the indice of the least RMSE
% RMSE(OptimalFunctionIndice) % show the specific value of the RMSE corresponding to the found indice
% for i = 1 : length(w_T)
%     subplot(13, 8, i); 
%     plot(T-273, w_T(:,i), ValidationData(:,1)-273, ValidationData(:,2),'ro','MarkerSize', 1.5); 
%     title(['G(\alpha) ' num2str(i) ' with RMSE ' num2str(RMSE(i))]);
% end
% % Plot the fitting performance of the optimal model identified
% figure
% plot(T-273, w_T(:,OptimalFunctionIndice), ValidationData(:,1)-273, ValidationData(:,2),'ro','MarkerSize', 1.5);
% title(['G(\alpha) ' num2str(OptimalFunctionIndice) ' with RMSE ' num2str(RMSE(OptimalFunctionIndice))]);
time_step(:, 4) = toc;
fprintf(['The time elapsed in Step FOUR was ' num2str(time_step(:, 4)) ' seconds.\n']) % show time elapsed in this step
 %% Step FIVE Calculate external heat required to drive the occurence of thermochemical reactions during pyrolysis
tic;
% Calculate the simplest formula of organic fraction in sludge sample
mole_C = X_new(1)/12; mole_H = X_new(2)/1; mole_O = X_new(3)/16; mole_N = X_new(4)/14; mole_S = X_new(5)/32;
fprintf(['The simplest formula of sludge sample is C1' 'H' num2str(mole_H/mole_C) 'O' num2str(mole_O/mole_C) 'N' num2str(mole_N/mole_C) 'S' num2str(mole_S/mole_C) '.\n']);
Mass_weight = 12 *1 + 1 * mole_H/mole_C + 16 * mole_O/mole_C + 14 * mole_N/mole_C + 32 * mole_S/mole_C;
EquivMole = 1000/Mass_weight; % the equivalent mole number of 1 kg dried sludge sample
% Calculate the need of enthalpy as per 1 kg dried sludge sample
Q_required = Enthalpy(T, Ea_new)/3.6E6 * EquivMole / 1; % in kWh/kg dried sludge
figure
plot(T-273, Q_required, 'ro');title('External energy required', 'FontSize', 14); xlabel('Pyrolysis temperature/^oC', 'FontSize', 12); ylabel(' Electricty required for pyrolysis process/(kWh/kg dried sludge)', 'FontSize', 12);
time_step(:, 5) = toc;
fprintf(['The time elapsed in Step FIVE was ' num2str(time_step(:, 5)) ' seconds.\n']) % show time elapsed in this step
%% Output the total time elapsed for the whole code
total_CPUtime = sum(time_step)/60 % in min, only for reference