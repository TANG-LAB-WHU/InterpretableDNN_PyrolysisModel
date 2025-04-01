% Script to extract variables from Results_trained.mat for SHAP analysis
fprintf('Loading Results_trained.mat and extracting variables for SHAP analysis...\n');

% Load the results from the main script - check multiple possible locations
try
    if exist('Results_trained.mat', 'file')
        fprintf('Found Results_trained.mat in current directory.\n');
        load('Results_trained.mat');
        
    elseif exist('output/results/Results_trained.mat', 'file')
        fprintf('Found Results_trained.mat in output/results directory.\n');
        load('output/results/Results_trained.mat');
        
    elseif exist('output/shap_analysis/Results_trained.mat', 'file')
        fprintf('Found Results_trained.mat in output/shap_analysis directory.\n');
        load('output/shap_analysis/Results_trained.mat');
        
    elseif exist('src/matlab/Results_trained.mat', 'file')
        fprintf('Found Results_trained.mat in src/matlab directory.\n');
        load('src/matlab/Results_trained.mat');
        
    elseif exist('output_shap_analysis/Results_trained.mat', 'file')
        fprintf('Found Results_trained.mat in output_shap_analysis directory.\n');
        load('output_shap_analysis/Results_trained.mat');
        
    else
        error('Results_trained.mat file not found in any expected directory.');
    end
catch err
    fprintf('Error loading Results_trained.mat: %s\n', err.message);
    error('Failed to load Results_trained.mat.');
end

% Create output directory if it doesn't exist
if ~exist('output/shap_analysis', 'dir')
    mkdir('output/shap_analysis');
end

% Prepare and extract variables for SHAP analysis
X = input;
Y = target;
XTest = input(:, testInd);
YTest = target(:, testInd);

% Save the required variables for SHAP analysis
fprintf('Saving variables for SHAP analysis...\n');
save('output/shap_analysis/trained_model.mat', 'net', 'trainInd', 'valInd', 'testInd', 'X', 'Y', 'XTest', 'YTest');

fprintf('Variables successfully extracted and saved for SHAP analysis.\n'); 