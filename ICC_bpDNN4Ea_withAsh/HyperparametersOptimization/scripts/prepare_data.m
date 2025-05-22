% Data Preparation Script for BP-DNN Hyperparameter Optimization
% This script prepares the input and target data for the BP-DNN model
% It uses the files from matlab_src as the starting point

% Load feedstock encoding data
fprintf('Loading feedstock data...\n');
load(fullfile(pwd, 'matlab_src', 'CopyrolysisFeedstock.mat'));

% Save a copy to the data directory
if ~exist(fullfile(pwd, 'data'), 'dir')
    mkdir(fullfile(pwd, 'data'));
end
save(fullfile(pwd, 'data', 'CopyrolysisFeedstock.mat'), ...
    'CopyrolysisFeedstockTag', 'Total_FeedID', 'Total_MixingRatio', ...
    'FeedstockIndex', 'Ea');

% Process feedstock data
fprintf('Processing feedstock data...\n');
FeedType_size = size(CopyrolysisFeedstockTag, 1);
Total_FeedID_size = size(Total_FeedID, 1);
Total_MixingRatio_size = size(Total_MixingRatio, 1);

MixingFeedID = zeros(Total_FeedID_size, FeedType_size);
MixingRatio = zeros(Total_MixingRatio_size, FeedType_size);

for i = 1 : Total_MixingRatio_size
    MixingFeedID(i, FeedstockIndex(i, 1)) = Total_FeedID(i, 1);
    MixingFeedID(i, FeedstockIndex(i, 2)) = Total_FeedID(i, 2);
    MixingRatio(i, FeedstockIndex(i, 1)) = Total_MixingRatio(i, 1);
    MixingRatio(i, FeedstockIndex(i, 2)) = Total_MixingRatio(i, 2);
end

Feedstock4training = [MixingFeedID MixingRatio];

% Load raw input data
fprintf('Loading raw input data...\n');
PreparedInputData = readmatrix(fullfile(pwd, 'matlab_src', 'RawInputData.xlsx'));

% Extract input features
Location = PreparedInputData(:, 1); 
VolatileMatters = PreparedInputData(:, 2);
FixedCarbon = PreparedInputData(:, 3);
Ash = PreparedInputData(:, 4);
C = PreparedInputData(:, 5);
H = PreparedInputData(:, 6);
N = PreparedInputData(:, 7);
O = PreparedInputData(:, 8);
S = PreparedInputData(:, 9);
Ash_SiO2 = PreparedInputData(:, 10);
Ash_Na2O = PreparedInputData(:, 11);
Ash_MgO = PreparedInputData(:, 12);
Ash_Al2O3 = PreparedInputData(:, 13);
Ash_K2O = PreparedInputData(:, 14);
Ash_CaO = PreparedInputData(:, 15);
Ash_P2O5 = PreparedInputData(:, 16);
Ash_CuO = PreparedInputData(:, 17);
Ash_ZnO = PreparedInputData(:, 18);
Ash_Fe2O3 = PreparedInputData(:, 19);
Degree_conversion = PreparedInputData(:, 20);

% Combine all input variables
Variables = [Location VolatileMatters FixedCarbon Ash C H N O S...
    Ash_SiO2 Ash_Na2O Ash_MgO Ash_Al2O3 Ash_K2O Ash_CaO Ash_P2O5 Ash_CuO Ash_ZnO Ash_Fe2O3...
    Degree_conversion Feedstock4training];

% Extract target data
Ea = PreparedInputData(:, 21);

% Create table for clearer organization
InputTable = array2table(Variables, 'VariableNames', {...
    'Location', 'VolatileMatters', 'FixedCarbon', 'Ash', 'C', 'H', 'N', 'O', 'S', ...
    'Ash_SiO2', 'Ash_Na2O', 'Ash_MgO', 'Ash_Al2O3', 'Ash_K2O', 'Ash_CaO', ...
    'Ash_P2O5', 'Ash_CuO', 'Ash_ZnO', 'Ash_Fe2O3', 'Degree_conversion'});

% Add feedstock columns
for i = 1:size(Feedstock4training, 2)
    if i <= FeedType_size
        InputTable.(['FeedID_' num2str(i)]) = Feedstock4training(:, i);
    else
        InputTable.(['MixingRatio_' num2str(i-FeedType_size)]) = Feedstock4training(:, i);
    end
end

% Create target table
TargetTable = table(Ea, 'VariableNames', {'Ea_kJ_mol'});

% Save prepared data
fprintf('Saving prepared data...\n');
save(fullfile(pwd, 'data', 'prepared_data.mat'), 'InputTable', 'TargetTable', 'Variables', 'Ea');
writetable(InputTable, fullfile(pwd, 'data', 'input_data.csv'));
writetable(TargetTable, fullfile(pwd, 'data', 'target_data.csv'));

% Run initial model using matlab_src scripts to establish baseline
fprintf('Running initial model using matlab_src scripts to establish baseline...\n');
try
    % Create baseline figures directory
    baseline_dir = fullfile(pwd, 'data', 'baseline');
    if ~exist(baseline_dir, 'dir')
        mkdir(baseline_dir);
    end
    baseline_figures_dir = fullfile(baseline_dir, 'Figures');
    if ~exist(baseline_figures_dir, 'dir')
        mkdir(baseline_figures_dir);
    end
    
    % Run bpDNN4Ea.m from matlab_src if it exists
    if exist(fullfile(pwd, 'matlab_src', 'bpDNN4Ea.m'), 'file')
        fprintf('Running bpDNN4Ea.m to establish baseline performance...\n');
        current_dir = pwd;
        cd(fullfile(pwd, 'matlab_src'));
        bpDNN4Ea;  % Run the script
        cd(current_dir);
        
        % Copy generated figures to baseline directory
        src_figures = dir(fullfile(pwd, 'matlab_src', 'Figures', '*.fig'));
        for i = 1:length(src_figures)
            copyfile(fullfile(pwd, 'matlab_src', 'Figures', src_figures(i).name), ...
                     fullfile(baseline_figures_dir, src_figures(i).name));
            fprintf('Copied baseline figure: %s\n', src_figures(i).name);
        end
        src_tifs = dir(fullfile(pwd, 'matlab_src', 'Figures', '*.tif'));
        for i = 1:length(src_tifs)
            copyfile(fullfile(pwd, 'matlab_src', 'Figures', src_tifs(i).name), ...
                     fullfile(baseline_figures_dir, src_tifs(i).name));
            fprintf('Copied baseline tif: %s\n', src_tifs(i).name);
        end
        
        % Copy Results_trained.mat if it exists
        if exist(fullfile(pwd, 'matlab_src', 'Results_trained.mat'), 'file')
            copyfile(fullfile(pwd, 'matlab_src', 'Results_trained.mat'), ...
                     fullfile(baseline_dir, 'Results_trained.mat'));
            fprintf('Copied baseline training results\n');
        end
    else
        fprintf('Warning: bpDNN4Ea.m not found in matlab_src. Skipping baseline model.\n');
    end
catch ME
    fprintf('Error running baseline model: %s\n', ME.message);
    fprintf('Continuing with hyperparameter optimization...\n');
end

% Copy necessary MATLAB functions from matlab_src to ensure they're available
fprintf('Copying necessary MATLAB functions...\n');
src_files = dir(fullfile(pwd, 'matlab_src', '*.m'));
for i = 1:length(src_files)
    copyfile(fullfile(pwd, 'matlab_src', src_files(i).name), ...
             fullfile(pwd, 'scripts', src_files(i).name));
    fprintf('Copied %s\n', src_files(i).name);
end

fprintf('Data preparation completed successfully.\n');

fprintf('Data preparation completed.\n');
