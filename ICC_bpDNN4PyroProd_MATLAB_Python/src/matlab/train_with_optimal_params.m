%% Training with Optimized Parameters
% This script trains the neural network model using the best parameters
% found during the optimization process. It loads the optimized parameters
% from the output/optimization_results/best_model.mat file and runs the full
% training process with these parameters.

clc;
fprintf('Starting model training with optimized parameters...\n');

%% Check if optimization results exist
if ~exist('output/optimization_results/best_model.mat', 'file')
    error(['Optimization results not found. ' ...
           'Please run optimize_model_params.m first to find the best parameters.']);
end

%% Load the best parameters
fprintf('Loading optimized parameters...\n');
load('output/optimization_results/best_model.mat', 'best_params_struct');

% Extract parameters
hidden_layers = best_params_struct.hidden_layers;
learning_rate = best_params_struct.learning_rate;
momentum = best_params_struct.momentum;
transfer_fcn = best_params_struct.transfer_fcn;

fprintf('Optimized parameters loaded:\n');
fprintf('  Hidden Layers: %s\n', mat2str(hidden_layers));
fprintf('  Learning Rate: %.3f\n', learning_rate);
fprintf('  Momentum: %.3f\n', momentum);
fprintf('  Transfer Function: %s\n\n', transfer_fcn);

%% Load and prepare data (same as in bpDNN4PyroProd.m)
fprintf('Loading and preparing data...\n');

% Load feedstock encoding data
load('CopyrolysisFeedstock.mat');

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

% Generate input and target dataset
PreparedInputData = readmatrix('RawInputData.xlsx');

% Input data
Location =  PreparedInputData(:, 1); 
VolatileMatters =  PreparedInputData(:, 2); FixedCarbon =  PreparedInputData(:, 3); Ash =  PreparedInputData(:, 4);
C =  PreparedInputData(:, 5); H =  PreparedInputData(:, 6); N =  PreparedInputData(:, 7); O =  PreparedInputData(:, 8); S =  PreparedInputData(:, 9);
TargetTemperature =  PreparedInputData(:, 10); ReactionTime =  PreparedInputData(:, 11); HeatingRate =  PreparedInputData(:, 12);
ReactorType =  PreparedInputData(:, 13);
Variables0 = [Location VolatileMatters FixedCarbon Ash C H N O S TargetTemperature ReactionTime...
    HeatingRate ReactorType];
Variables = [Variables0 Feedstock4training];

% Target data
CharYield = PreparedInputData(:, 14);
LiquidYield = PreparedInputData(:, 15);
GasYield = PreparedInputData(:, 16);
ProductsYield = [CharYield LiquidYield GasYield];

% Name Variables as RESPONSE and PREDICTOR
Target = table(ProductsYield(:, 1), ProductsYield(:, 2), ProductsYield(:, 3));
Target.Properties.VariableNames([1 2 3]) = {'Char/%' 'Liquid/%' 'Gas/%'};

% Assign training dataset
input = Variables'; 
target = Target.Variables';

%% Configure and train the neural network with optimized parameters
fprintf('Configuring neural network with optimized parameters...\n');

% Create network with optimized hidden layers
net = nncreate(hidden_layers);

% Set optimized training parameters
net.trainParam.mc = momentum;
net.trainParam.lr = learning_rate;
net.trainParam.lr_inc = 0.15;
net.trainParam.lr_dec = 1.40;
net.trainParam.goal = 1e-6;
net.trainParam.min_grad = 1e-5;
net.trainParam.max_fail = 20;
net.trainParam.epoch = 6000;  % Using full epochs for final training

% Set transfer functions using optimized transfer function
Nl = size(hidden_layers, 2) + 1;
for i = 1 : Nl
    if i == Nl
        net.layer{i}.transferFcn = 'purelin'; % Output layer always uses purelin
    else
        net.layer{i}.transferFcn = transfer_fcn; % Hidden layers use optimized function
    end
end

% Normalization
net.processFcn = 'mapminmax';
net.processParam.ymax = 1;
net.processParam.ymin = -1;

% Partition the dataset for training, validation, and testing purposes
net.divideFcn = 'dividerand';
net.divideParam.trainRatio = 0.80;
net.divideParam.valRatio = 0.10;
net.divideParam.testRatio = 0.10;

% Set up the real-time display for overseeing learning process
net.performFcn = 'mse';
net.adaptFcn = 'none';
net.trainParam.showCommandLine = true;
net.trainParam.showWindow = true;

%% Train the network with optimized parameters
fprintf('Training network with optimized parameters...\n');
[net, tr] = nntrain(net, input, target);

%% Process results (similar to bpDNN4PyroProd.m)
global PS TS

trainInd = tr.trainInd;
valInd = tr.valInd;
testInd = tr.testInd;

validation = true;
testing =  true;

if isempty(valInd), validation  = false; end
if isempty(testInd), testing  = false; end

trainInp = input(:, trainInd);
trainTarg = target(:, trainInd);
trainOut = nnpredict(net, trainInp);

if validation
    valInp = input(:, valInd);
    valTarg = target(:, valInd);
    valOut = nnpredict(net, valInp);
end

if testing
    testInp = input(:, testInd);
    testTarg = target(:, testInd);
    testOut = nnpredict(net, testInp);
end

trainPerf = tr.best_perf;
valPerf = tr.best_vperf;
testPerf = tr.best_tperf;

%% Create visualization (similar to bpDNN4PyroProd.m)
% Plot the training performance
trainPerformance = nneval(net, trainInp, trainTarg);
figure('Name',['Whole training performance in MSE = ' num2str(trainPerformance)])
subplot(3, 1, 1)
plot(trainInd, trainTarg(1, :), 'r-o', trainInd, trainOut(1, :), 'b-.s'); 
legend({'Observed char yield (%)', 'Predicted char yield (%)'}, 'Location','northwest'); xlabel('Index in dataset'); ylabel('Output in training');
subplot(3, 1, 2)
plot(trainInd, trainTarg(2, :), 'r-o', trainInd, trainOut(2, :), 'b-.s'); 
legend({'Observed liquid yield (%)', 'Predicted liquid yield (%)'}, 'Location','northwest'); xlabel('Index in dataset'); ylabel('Output in training');
subplot(3, 1, 3)
plot(trainInd, trainTarg(3, :), 'r-o', trainInd, trainOut(3, :), 'b-.s'); 
legend({'Observed gas yield (%)', 'Predicted gas yield (%)'}, 'Location','northwest'); xlabel('Index in dataset'); ylabel('Output in training');

% Plot the validation performance
if validation
    valPerformance = nneval(net, valInp, valTarg);
    figure('Name',['Whole validation performance in MSE = ' num2str(valPerformance)])
    subplot(3, 1, 1)
    plot(valInd, valTarg(1, :), 'r-o', valInd, valOut(1, :), 'b-.s'); 
    legend({'Observed char yield (%)', 'Predicted char yield (%)'}, 'Location','northwest'); xlabel('Index in dataset'); ylabel('Output in validation');
    subplot(3, 1, 2)
    plot(valInd, valTarg(2, :), 'r-o', valInd, valOut(2, :), 'b-.s'); 
    legend({'Observed liquid yield (%)', 'Predicted liquid yield (%)'}, 'Location','northwest'); xlabel('Index in dataset'); ylabel('Output in validation');
    subplot(3, 1, 3)
    plot(valInd, valTarg(3, :), 'r-o', valInd, valOut(3, :), 'b-.s'); 
    legend({'Observed gas yield (%)', 'Predicted gas yield (%)'}, 'Location','northwest'); xlabel('Index in dataset'); ylabel('Output in validation');
end

% Plot the testing performance
if testing
    testPerformance = nneval(net, testInp, testTarg);
    figure('Name', ['Whole testing performance in MSE = ' num2str(testPerformance)])
    subplot(3, 1, 1)
    plot(testInd, testTarg(1, :), 'r-o', testInd, testOut(1, :), 'b-.s'); 
    legend({'Observed char yield (%)', 'Predicted char yield (%)'}, 'Location','northwest'); xlabel('Index in dataset'); ylabel('Output in testing');
    subplot(3, 1, 2)
    plot(testInd, testTarg(2, :), 'r-o', testInd, testOut(2, :), 'b-.s'); 
    legend({'Observed liquid yield (%)', 'Predicted liquid yield (%)'}, 'Location','northwest'); xlabel('Index in dataset'); ylabel('Output in testing');
    subplot(3, 1, 3)
    plot(testInd, testTarg(3, :), 'r-o', testInd, testOut(3, :), 'b-.s'); 
    legend({'Observed gas yield (%)', 'Predicted gas yield (%)'}, 'Location','northwest'); xlabel('Index in dataset'); ylabel('Output in testing');
end

% Plot the regression performance
figure
% For overall case
subplot(2, 2, 1)
Overall_obs = [trainTarg valTarg testTarg];
Overall_pred = [trainOut valOut testOut];
plot(Overall_obs, Overall_pred,'ro','MarkerSize', 4); hold on
maximum = max(Overall_obs(:)); minimum = min(Overall_obs(:));
x_overall = linspace(minimum, maximum, length(Overall_obs));
plot(x_overall, x_overall, 'b');
Rsq_overall = 1 - sum((Overall_obs - Overall_pred).^2)/sum((Overall_obs - mean(Overall_obs)).^2);
title(['Overall performance R^2 = ' num2str(Rsq_overall)]); xlabel('Observed values in yield (%)'); ylabel('Predicted values in yield (%)');

% For training case
subplot(2, 2, 2)
plot(trainTarg, trainOut,'ro','MarkerSize', 4); hold on
maximum = max(trainTarg(:)); minimum = min(trainTarg(:));
x_train = linspace(minimum, maximum, length(trainTarg));
plot(x_train, x_train, 'b');
Rsq_training = 1 - sum((trainTarg - trainOut).^2)/sum((trainTarg - mean(trainTarg)).^2);
title(['Training performance R^2 = ' num2str(Rsq_training)]); xlabel('Observed values in yield (%)'); ylabel('Predicted values in yield (%)');

% For validation case
subplot(2, 2, 3)
plot(valTarg, valOut,'ro','MarkerSize', 4); hold on
maximum = max(valTarg(:)); minimum = min(valTarg(:));
x_val = linspace(minimum, maximum, length(valTarg));
plot(x_val, x_val, 'b');
Rsq_validation = 1 - sum((valTarg - valOut).^2)/sum((valTarg - mean(valTarg)).^2);
title(['Validation performance R^2 = ' num2str(Rsq_validation)]); xlabel('Observed values in yield (%)'); ylabel('Predicted values in yield (%)');

% For testing case
subplot(2, 2, 4)
plot(testTarg, testOut,'ro','MarkerSize', 4); hold on
maximum = max(testTarg(:)); minimum = min(testTarg(:));
x_test = linspace(minimum, maximum, length(testTarg));
plot(x_test, x_test, 'b');
Rsq_testing = 1 - sum((testTarg - testOut).^2)/sum((testTarg - mean(testTarg)).^2);
title(['Testing performance R^2 = ' num2str(Rsq_testing)]); xlabel('Observed values in yield (%)'); ylabel('Predicted values in yield (%)');

% Visualize weights
numHiddenLayer = length(hidden_layers);
for i = 1 : numHiddenLayer + 1
     if i == 1
         figure(i + 5)
         glyphplot(net.weight{1, i}); title('Weights from Input Layer to Hidden Layer 1');
     end
     if i > 1
         figure(i + 5)
         glyphplot(net.weight{1, i}); title(['Weights from Hidden Layer ' num2str(i-1) ' to Hidden Layer ' num2str(i)]);
     end
      if i == numHiddenLayer + 1
         figure(i + 5)
         glyphplot(net.weight{1, i}); title(['Weights from Hidden Layer ' num2str(i-1) ' to Output Layer']);
      end
end

%% Save figures and results
fprintf('Saving figures and results...\n');

% Create Figures directory
mkdir('Figures');
FolderName = 'Figures';
FigList = findobj(allchild(0), 'flat', 'Type', 'figure');
for j = 1:length(FigList)
  FigHandle = FigList(j);
  FigName = ['Fig' num2str(j)];
  savefig(FigHandle, fullfile(FolderName, [FigName '.fig']));
  saveas(FigHandle, fullfile(FolderName,  [FigName '.tif']));
end

% Ensure output directories exist
if ~exist('output/results', 'dir')
    fprintf('Creating output/results directory...\n');
    mkdir('output/results');
end

% Add optimization information to the results
fprintf('Adding optimization information to results...\n');
optimization_info = struct();
optimization_info.optimized_params = best_params_struct;
optimization_info.hidden_layers = hidden_layers;
optimization_info.learning_rate = learning_rate;
optimization_info.momentum = momentum;
optimization_info.transfer_fcn = transfer_fcn;

% Save results with optimization info
save('Results_trained.mat');

% Save results to output directory
copyfile('Results_trained.mat', 'output/results/Results_trained.mat');

% Copy figures to output directory
fprintf('Copying Figures to output/results/Figures...\n');
if ~exist('output/results/Figures', 'dir')
    mkdir('output/results/Figures');
end
copyfile('Figures/*', 'output/results/Figures/');

% Create performance report
fid = fopen('output/results/optimized_model_performance.txt', 'w');
fprintf(fid, 'Optimized Neural Network Model Performance\n');
fprintf(fid, '========================================\n\n');
fprintf(fid, 'Training completed on: %s\n\n', datestr(now));
fprintf(fid, 'Model Architecture:\n');
fprintf(fid, '  Hidden Layers: %s\n', mat2str(hidden_layers));
fprintf(fid, '  Transfer Function: %s\n\n', transfer_fcn);
fprintf(fid, 'Training Parameters:\n');
fprintf(fid, '  Learning Rate: %.3f\n', learning_rate);
fprintf(fid, '  Momentum: %.3f\n', momentum);
fprintf(fid, '  Max Epochs: %d\n\n', net.trainParam.epoch);
fprintf(fid, 'Performance Metrics:\n');
fprintf(fid, '  Overall R^2: %.6f\n', Rsq_overall);
fprintf(fid, '  Training R^2: %.6f\n', Rsq_training);
fprintf(fid, '  Validation R^2: %.6f\n', Rsq_validation);
fprintf(fid, '  Testing R^2: %.6f\n\n', Rsq_testing);
fprintf(fid, '  Training MSE: %.6f\n', trainPerformance);
if validation
    fprintf(fid, '  Validation MSE: %.6f\n', valPerformance);
end
if testing
    fprintf(fid, '  Testing MSE: %.6f\n', testPerformance);
end
fclose(fid);

% Create zip backup of all necessary files
zip('output/results/Result_needed.zip', {'Results_trained.mat','Figures', 'train_with_optimal_params.m'});

fprintf('\nTraining with optimal parameters completed.\n');
fprintf('Results saved to output/results/\n');
fprintf('To proceed with SHAP analysis, run extract_vars_for_shap.m\n');
fprintf('Analysis complete.\n'); 