%% Predict the pyrolytic product distribution for an unknown sample

% Load the trained bp-DNN model into the workspace
 
close all; clc;clear all

load('Results_trained.mat')

load SampleIn_pred
sampleInp = Pred_China_biosolids';

sampleOut = nnpredict(net, sampleInp)/100 % show the predicted yiled values of Char, Liquid, and Gas by column
AbsoluteSUM = sum(sampleOut) % see if the total of prodicted yield of the three pyrolytic products equalizes to be 1
RelativeError = abs((AbsoluteSUM - 1)./1*100) % With regard to the sum of 1

figure
Predicted_products = sampleOut' * 100;
PyrolysisTemperature = sampleInp(10, :)';
bar(PyrolysisTemperature, Predicted_products,"stacked");
xlabel('Pyrolysis temperature/^oC'); ylabel('Predicted pyrolytic products yield/%')
legend('Char', 'Liquid', 'Gas', 'Location', 'northoutside', 'NumColumns', 3);

figure
plot(PyrolysisTemperature, RelativeError, 'b-o');
xlabel('Pyrolysis temperature/^oC'); ylabel('Relative error of total yield of predicted pyrolytic products/%')