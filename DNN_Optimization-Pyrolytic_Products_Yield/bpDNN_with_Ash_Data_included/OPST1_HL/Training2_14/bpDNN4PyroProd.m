clc;clear all % clean workspace for new job

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
Ash_SiO2 = PreparedInputData(:, 10); Ash_Na2O = PreparedInputData(:, 11); Ash_MgO = PreparedInputData(:, 12); Ash_Al2O3 = PreparedInputData(:, 13);
    Ash_K2O = PreparedInputData(:, 14); Ash_CaO = PreparedInputData(:, 15); Ash_P2O5 = PreparedInputData(:, 16); Ash_CuO = PreparedInputData(:, 17);
    Ash_ZnO = PreparedInputData(:, 18); Ash_Fe2O3 = PreparedInputData(:, 19); 
TargetTemperature =  PreparedInputData(:, 20); ReactionTime =  PreparedInputData(:, 21); HeatingRate =  PreparedInputData(:, 22);
    ReactorType =  PreparedInputData(:, 23);
Variables0 = [Location VolatileMatters FixedCarbon Ash C H N O S...
    Ash_SiO2 Ash_Na2O Ash_MgO Ash_Al2O3 Ash_K2O Ash_CaO Ash_P2O5 Ash_CuO Ash_ZnO Ash_Fe2O3...
    TargetTemperature ReactionTime HeatingRate ReactorType];
Variables = [Variables0 Feedstock4training];

% Target data
CharYield = PreparedInputData(:, 24);
LiquidYield = PreparedInputData(:, 25);
GasYield = PreparedInputData(:, 26);
ProductsYield = [CharYield LiquidYield GasYield];

% Name Variables as RESPONSE and PREDICTOR
Target = table(ProductsYield(:, 1), ProductsYield(:, 2), ProductsYield(:, 3));
Target.Properties.VariableNames([1 2 3]) = {'Char/%' 'Liquid/%' 'Gas/%'};

% Assign training dataset
input = Variables'; % In Variables,  the name of each column was named by the order of {'Location' 'VolatileMatters/%'...
    % 'FixedCarbon/%' 'Ash/%' 'C/%' 'H/%' 'N/%' 'O/%' 'S/%'...
 % 'TargetTemperature/Celsius' 'ReactionTime/min' 'HeatingRate/(K/min)' 'ReactorType' 'FeedstockType'}
target = Target.Variables';

%% Settings of bp-DNN architecture
hiddenLayer = 14;

net = nncreate(hiddenLayer);

% Training parameters
net.trainParam.mc = 0.97; % Optimal net.trainParam.mc = 0.97
net.trainParam.lr = 0.85;
net.trainParam.lr_inc = 0.15;
net.trainParam.lr_dec = 1.40;% reference 1.40
net.trainParam.goal = 1e-6;
net.trainParam.min_grad = 1e-5;
net.trainParam.max_fail = 20;
net.trainParam.epoch = 5000;

Nl = size(hiddenLayer, 2) + 1;
% NI = net.numLayer;
for i = 1 : Nl
    if i == Nl
        net.layer{i}.transferFcn = 'purelin'; % the transfer function from the last hidden layer to the output layer, available ones including 'purelin', 'logsig', and 'softmax'
    else
        net.layer{i}.transferFcn = 'logsig'; % the transfer function in hidden layers, available ones including 'tansig', 'logsig', and 'tansigopt' (unavailable)
    end
end
% Transfer functions list used for training neural networks
%     compet - Competitive transfer function.
%     elliotsig - Elliot sigmoid transfer function.
%     hardlim - Positive hard limit transfer function.
%     hardlims - Symmetric hard limit transfer function.
%     logsig - Logarithmic sigmoid transfer function.
%     netinv - Inverse transfer function.
%     poslin - Positive linear transfer function.
%     purelin - Linear transfer function.
%     radbas - Radial basis transfer function.
%     radbasn - Radial basis normalized transfer function.
%     satlin - Positive saturating linear transfer function.
%     satlins - Symmetric saturating linear transfer function.
%     softmax - Soft max transfer function.
%     tansig - Symmetric sigmoid transfer function.
%     tribas - Triangular basis transfer function.

% Normalization
net.processFcn = 'mapminmax'; % mapminmax or mapstd
net.processParam.ymax = 1;
net.processParam.ymin = -1;

% Partition the dataset for training, validation, and testing purposes
net.divideFcn = 'dividerand'; % dividerand (using random indices or divideint (using interleaved indices) or 
% dividetrain (assigning all targets to the training set and no targets to the validation or test sets) or
% divideblock (dividing targets into three sets using blocks of indices)
net.divideParam.trainRatio = 0.90;%0.8 in reference case
net.divideParam.valRatio = 0.05;%0.1 in reference case
net.divideParam.testRatio = 0.05;%0.1 in reference case

% Set up the real-time display for overseeing learning process
net.performFcn = 'mse'; % mse, or mae, or sse, or sae
net.adaptFcn = 'none';
net.trainParam.showCommandLine = true;
net.trainParam.showWindow = true;
%% 

% Initialize the training for the desirable neural network
[net, tr] = nntrain(net, input, target);
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

% Plot the training performance in the output of triple products between the observed and predicted
trainPerformance = nneval(net, trainInp, trainTarg);
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
    valPerformance = nneval(net, valInp, valTarg);
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
    testPerformance = nneval(net, testInp, testTarg);
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
% clear all
% exit % automatically quit MATLAB after running