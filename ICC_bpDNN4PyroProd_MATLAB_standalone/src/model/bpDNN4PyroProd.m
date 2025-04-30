function [varargout] = bpDNN4PyroProd()
%% bpDNN4PyroProd - Neural Network for Pyrolysis Product Prediction
%
% This script performs complete neural network training for pyrolysis product prediction
% It can be used in two ways:
% 1. As a standalone script: Run directly to perform complete training and visualization
% 2. As a function: Call with output arguments to get prepared data matrices
%
% Usage as function:
%   [input_data, target_data, PS_global, TS_global] = bpDNN4PyroProd()
%
% Author: Original code author

% Check if script is being called as a function
isFunction = nargout > 0;

global PS TS

%% Prepare Input and Target data used for learning
% Convert feedstocks in mono-pyrolysis and co-pyrolysis to digitalized data
load('CopyrolysisFeedstock.mat') % load feedstock encoding data

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
input_data = Variables'; % In Variables, the name of each column was named by the order of {'Location' 'VolatileMatters/%'...
    % 'FixedCarbon/%' 'Ash/%' 'C/%' 'H/%' 'N/%' 'O/%' 'S/%'...
 % 'TargetTemperature/Celsius' 'ReactionTime/min' 'HeatingRate/(K/min)' 'ReactorType' 'FeedstockType'}
target_data = Target.Variables';

% Create preprocessing structures if not already defined
if ~exist('PS', 'var') || isempty(PS)
    PS = struct();
end

if ~exist('TS', 'var') || isempty(TS)
    TS = struct();
end

PS_global = PS;
TS_global = TS;

% If called as a function, return the prepared data and exit
if isFunction
    varargout{1} = input_data;
    varargout{2} = target_data;
    varargout{3} = PS_global;
    varargout{4} = TS_global;
    return;
end

% The rest of the code only runs when used as a script
% When called as a function, execution stops at the return statement above

%% Settings of bp-DNN architecture
hiddenLayer = [37, 37, 37, 37, 37];
net = nncreate(hiddenLayer);

% Training parameters
net.trainParam.mc = 0.97; % Optimal net.trainParam.mc = 0.97
net.trainParam.lr = 0.85; % A large learing rate (lr) could accelerate the learning speed for convergence. 0.75 for reference
net.trainParam.lr_inc = 0.15;
net.trainParam.lr_dec = 1.40;% reference 1.40
net.trainParam.goal = 1e-6;
net.trainParam.min_grad = 1e-5;
net.trainParam.max_fail = 20;
net.trainParam.epoch = 6000;

Nl = size(hiddenLayer, 2) + 1;
% NI = net.numLayer;
for i = 1 : Nl
    if i == Nl
        net.layer{i}.transferFcn = 'purelin'; % the transfer function from the last hidden layer to the output layer, available ones including 'purelin', 'logsig', and 'softmax'
    else
        net.layer{i}.transferFcn = 'logsig'; % the transfer function in hidden layers, available ones including 'tansig', 'logsig', and 'tansigopt' (unavailable)
    end
end
% Transfer functions list used for neural networks
%     compet    - Competitive transfer function (outputs 1 for max input, 0 elsewhere).
%     elliotsig - Elliott sigmoid transfer function (computationally efficient): f(x) = x/(1+|x|)
%     hardlim   - Hard limit transfer function (binary threshold): f(x) = 1 if x>=0, 0 otherwise
%     hardlims  - Symmetric hard limit function: f(x) = 1 if x>=0, -1 otherwise
%     logsig    - Logarithmic sigmoid: f(x) = 1/(1+e^(-x)), range [0,1]
%     netinv    - Inverse function: f(x) = 1/x
%     poslin    - Positive linear (ReLU): f(x) = max(0,x)
%     purelin   - Pure linear: f(x) = x, for regression problems
%     radbas    - Radial basis function: f(x) = e^(-x^2)
%     radbasn   - Normalized radial basis function
%     satlin    - Saturating linear: f(x) = 0 if x<0, x if 0<=x<=1, 1 if x>1
%     satlins   - Symmetric saturating linear: f(x) = -1 if x<-1, x if -1<=x<=1, 1 if x>1
%     softmax   - Softmax transfer function: f(x_i) = e^(x_i)/sum(e^(x_j)), probabilistic output
%     tansig    - Hyperbolic tangent sigmoid: f(x) = 2/(1+e^(-2x))-1, range [-1,1]
%     tribas    - Triangular basis function: f(x) = 1-|x| if -1<=x<=1, 0 otherwise
%
% Recommended combinations:
% 1. For regression problems:
%    - Hidden layers: 'tansig' or 'logsig'
%    - Output layer: 'purelin'
% 2. For binary classification:
%    - Hidden layers: 'tansig' or 'logsig'
%    - Output layer: 'logsig'
% 3. For multi-class classification:
%    - Hidden layers: 'tansig' or 'logsig'
%    - Output layer: 'softmax'
% 4. For function approximation with ReLU-based networks:
%    - Hidden layers: 'poslin'
%    - Output layer: 'purelin'

% Normalization
net.processFcn = 'mapminmax'; % mapminmax or mapstd
net.processParam.ymax = 1;
net.processParam.ymin = -1;

% Partition the dataset for training, validation, and testing purposes
net.divideFcn = 'dividerand'; % dividerand (using random indices or divideint (using interleaved indices) or 
% dividetrain (assigning all targets to the training set and no targets to the validation or test sets) or
% divideblock (dividing targets into three sets using blocks of indices)
net.divideParam.trainRatio = 0.80;%0.8 in reference case
net.divideParam.valRatio = 0.10;%0.1 in reference case
net.divideParam.testRatio = 0.10;%0.1 in reference case

% Set up the real-time display for overseeing learning process
net.performFcn = 'mse'; % mse, or mae, or sse, or sae
net.adaptFcn = 'none';
net.trainParam.showCommandLine = true;
net.trainParam.showWindow = true;
%% 

% Initialize the training for the desirable neural network
[net, tr] = nntrain(net, input_data, target_data);

trainInd = tr.trainInd;
valInd = tr.valInd;
testInd = tr.testInd;

% Check which strategy was used (TT or TVT)
validation = ~isempty(valInd);
testing = ~isempty(testInd);

if ~validation
    fprintf('Training using TT strategy (no validation set)\n');
else
    fprintf('Training using TVT strategy\n');
end

trainInp = input_data(:, trainInd);
trainTarg = target_data(:, trainInd);
trainOut = nnpredict(net, trainInp);

if validation
    valInp = input_data(:, valInd);
    valTarg = target_data(:, valInd);
    valOut = nnpredict(net, valInp);
end

if testing
    testInp = input_data(:, testInd);
    testTarg = target_data(:, testInd);
    testOut = nnpredict(net, testInp);
end

% Performance metrics
trainPerf = tr.best_perf;
if validation
    valPerf = tr.best_vperf;
else
    valPerf = NaN;
end
if testing
    testPerf = tr.best_tperf;
else
    testPerf = NaN;
end

% Plot the training performance in the output of triple products between the observed and predicted
try
    [trainPerformance, ~] = nneval(net, trainInp, trainTarg);
catch ME
    if ~isempty(ME.identifier)
        warning(ME.identifier, '%s', ME.message);
    else
        warning('NeuralNet:EvalError', '%s', ME.message);
    end
    trainPerformance = NaN;
end
figure('Name',['Whole training perforance in MSE = ' num2str(trainPerformance)])
% index_train = 1 : length(trainInd);
subplot(3, 1, 1)
plot(trainInd, trainTarg(1, :), 'r-o', trainInd, trainOut(1, :), 'b-.s'); 
legend({'Observed char yield (%)', 'Predicted char yield (%)'}, 'Location','northwest'); xlabel('Index in dataset'); ylabel('Output in training');
subplot(3, 1, 2)
plot(trainInd, trainTarg(2, :), 'r-o', trainInd, trainOut(2, :), 'b-.s'); 
legend({'Observed liquid yield (%)', 'Predicted liquid yield (%)'}, 'Location','northwest'); xlabel('Index in dataset'); ylabel('Output in training');
subplot(3, 1, 3)
plot(trainInd, trainTarg(3, :), 'r-o', trainInd, trainOut(3, :), 'b-.s'); 
legend({'Observed gas yield (%)', 'Predicted gas yield (%)'}, 'Location','northwest'); xlabel('Index in dataset'); ylabel('Output in training');

% Plot the validation performance in the output of triple products between the observed and predicted
if validation
    try
        [valPerformance, ~] = nneval(net, valInp, valTarg);
    catch ME
        if ~isempty(ME.identifier)
            warning(ME.identifier, '%s', ME.message);
        else
            warning('NeuralNet:EvalError', '%s', ME.message);
        end
        valPerformance = NaN;
    end
    figure('Name',['Whole validation perforance in MSE = ' num2str(valPerformance)])
    % index_val = 1 : length(valInd);
    subplot(3, 1, 1)
    % plot(index_val, valTarg(1, :), 'r-o', index_val, valOut(1, :), 'b-.s'); 
    plot(valInd, valTarg(1, :), 'r-o', valInd, valOut(1, :), 'b-.s'); 
    legend({'Observed char yield (%)', 'Predicted char yield (%)'}, 'Location','northwest'); xlabel('Index in dataset'); ylabel('Outputin in validation');
    subplot(3, 1, 2)
    %plot(index_val, valTarg(2, :), 'r-o', index_val, valOut(2, :), 'b-.s'); 
    plot(valInd, valTarg(2, :), 'r-o', valInd, valOut(2, :), 'b-.s'); 
    legend({'Observed liquid yield (%)', 'Predicted liquid yield (%)'}, 'Location','northwest'); xlabel('Index in dataset'); ylabel('Output in validation');
    subplot(3, 1, 3)
    % plot(index_val, valTarg(3, :), 'r-o', index_val, valOut(3, :), 'b-.s');
    plot(valInd, valTarg(3, :), 'r-o', valInd, valOut(3, :), 'b-.s'); 
    legend({'Observed gas yield (%)', 'Predicted gas yield (%)'}, 'Location','northwest'); xlabel('Index in dataset'); ylabel('Output in validation');
end

% Plot the testing performance in the output of triple products between the observed and predicted
if testing
    try
        [testPerformance, ~] = nneval(net, testInp, testTarg);
    catch ME
        if ~isempty(ME.identifier)
            warning(ME.identifier, '%s', ME.message);
        else
            warning('NeuralNet:EvalError', '%s', ME.message);
        end
        testPerformance = NaN;
    end
    figure('Name', ['Whole testing perforance in MSE = ' num2str(testPerformance)])
    % index_test = 1 : length(testInd);
    subplot(3, 1, 1)
    % plot(index_test, testTarg(1, :), 'r-o', index_test, testOut(1, :), 'b-.s'); 
    plot(testInd, testTarg(1, :), 'r-o', testInd, testOut(1, :), 'b-.s'); 
    legend({'Observed char yield (%)', 'Predicted char yield (%)'}, 'Location','northwest'); xlabel('Index in dataset'); ylabel('Output in testing');
    subplot(3, 1, 2)
    % plot(index_test, testTarg(2, :), 'r-o', index_test, testOut(2, :), 'b-.s'); 
    plot(testInd, testTarg(2, :), 'r-o', testInd, testOut(2, :), 'b-.s'); 
    legend({'Observed liquid yield (%)', 'Predicted liquid yield (%)'}, 'Location','northwest'); xlabel('Index in dataset'); ylabel('Output in testing');
    subplot(3, 1, 3)
    %plot(index_test, testTarg(3, :), 'r-o', index_test, testOut(3, :), 'b-.s'); 
    plot(testInd, testTarg(3, :), 'r-o', testInd, testOut(3, :), 'b-.s'); 
    legend({'Observed gas yield (%)', 'Predicted gas yield (%)'}, 'Location','northwest'); xlabel('Index in dataset'); ylabel('Output in testing');
end

% Plot the regression performance of training, validation, testing, and the overall bteween the obserables and the predicted ones
figure
% For overall case
subplot(2, 2, 1)
if validation && testing
    Overall_obs = [trainTarg valTarg testTarg];
    Overall_pred = [trainOut valOut testOut];
elseif testing && ~validation
    Overall_obs = [trainTarg testTarg];
    Overall_pred = [trainOut testOut];
else
    Overall_obs = trainTarg;
    Overall_pred = trainOut;
end
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

% For validation case (only if using TVT strategy)
if validation
    subplot(2, 2, 3)
    plot(valTarg, valOut,'ro','MarkerSize', 4); hold on
    maximum = max(valTarg(:)); minimum = min(valTarg(:));
    x_val = linspace(minimum, maximum, length(valTarg));
    plot(x_val, x_val, 'b');
    Rsq_validation = 1 - sum((valTarg - valOut).^2)/sum((valTarg - mean(valTarg)).^2);
    title(['Validation performance R^2 = ' num2str(Rsq_validation)]); xlabel('Observed values in yield (%)'); ylabel('Predicted values in yield (%)');
else
    % For TT strategy, show a blank or message in the validation subplot
    subplot(2, 2, 3)
    text(0.5, 0.5, 'No Validation Set (TT Strategy)', 'HorizontalAlignment', 'center');
    axis off;
end

% For testing case
if testing
    subplot(2, 2, 4)
    plot(testTarg, testOut,'ro','MarkerSize', 4); hold on
    maximum = max(testTarg(:)); minimum = min(testTarg(:));
    x_test = linspace(minimum, maximum, length(testTarg));
    plot(x_test, x_test, 'b');
    Rsq_testing = 1 - sum((testTarg - testOut).^2)/sum((testTarg - mean(testTarg)).^2);
    title(['Testing performance R^2 = ' num2str(Rsq_testing)]); xlabel('Observed values in yield (%)'); ylabel('Predicted values in yield (%)');
else
    % If no testing set exists
    subplot(2, 2, 4)
    text(0.5, 0.5, 'No Testing Set', 'HorizontalAlignment', 'center');
    axis off;
end

% Visualize weights between Input, Hidden, and Output layers
numHiddenLayer = length(hiddenLayer);
% figure
for i = 1 : numHiddenLayer + 1
     if i == 1
         figure(i + 5) % prevent the overwriting of the previously plotted Figures by increasing the indice of the new figure
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

mkdir('Figures');   % creat a dedicated folder to save all the output figures
FolderName = 'Figures';
FigList = findobj(allchild(0), 'flat', 'Type', 'figure');
for j = 1:length(FigList)
  FigHandle = FigList(j);
  FigName = ['Fig' num2str(j)];
  savefig(FigHandle, fullfile(FolderName, [FigName '.fig']));
  saveas(FigHandle, fullfile(FolderName,  [FigName '.tif']));
end

save("Results_trained.mat");% save the output data results after running
zip('Result_needed.zip', {'Results_trained.mat','Figures', 'bpDNN4PyroProd.m'});
% rmdir Figures s
% delete('Results_trained.mat')

close all; clc; 
%clear all
% exit % automatically quit MATLAB after running
end