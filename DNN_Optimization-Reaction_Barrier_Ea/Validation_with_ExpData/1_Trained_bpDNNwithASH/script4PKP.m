%% Predict the pyrolytic product distribution for an unknown sample

% Load the trained bp-DNN model into the workspace
 
clc;clear all

% Load trained neural network model
load('Results_trained.mat')

% Load validation data
load 1202Validation_WoodSawdust.mat
load 1202Validation_SewageSludge.mat

% Prepare input data for prediction
sampleInp_WS = newSawdust';
sampleInp_SS= newSewageSludge';

% Predict activation energy using trained model
SampleOut_WoodSawdust = nnpredict(net, sampleInp_WS) .* 1000; % show the predicted value of Ea in J/mol
SampleOut_SewageSludge = nnpredict(net, sampleInp_SS) .* 1000; % show the predicted value of Ea in J/mol

close all;

% Get current directory to handle path correctly
current_dir = pwd;
parent_dir = fileparts(current_dir);

% Save the prediction results to be used in the next step with full path
save(fullfile(parent_dir, 'Ea_prediction_results.mat'), 'SampleOut_SewageSludge', 'SampleOut_WoodSawdust');

% Display results for verification
fprintf('Prediction completed. Results saved to %s\n', fullfile(parent_dir, 'Ea_prediction_results.mat'));
fprintf('Predicted activation energy for sewage sludge: %.2f J/mol\n', mean(SampleOut_SewageSludge));

clearvars -except SampleOut_SewageSludge SampleOut_WoodSawdust; % specifically save the needed outputs while delete the others