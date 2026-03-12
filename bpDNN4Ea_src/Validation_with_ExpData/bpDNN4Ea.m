% Load prepared dataset for training
load dataset4Ea

% Name Variables as RESPONSE and PREDICTOR
Input = table(Variables(:,1), Variables(:,2), Variables(:,3), Variables(:,4),...
    Variables(:,5), Variables(:,6), Variables(:,7), Variables(:,8));
Input.Properties.VariableNames([1 2 3 4 5 6 7 8]) = {'C/%' 'H/%' 'O/%' 'N/%' 'S/%' 'VolatileMatters/%' ...
    'FixedCarbon/%' 'Ash/%'};

Target = table(Ea(:, 1));
Target.Properties.VariableNames(1) = {'Ea/kJ/mol'};

% Assign training dataset
input = Input.Variables';
target = Target.Variables';

% Settings of bp-DNN architecture
hiddenLayerSize = [7, 7, 7];
net = nncreate(hiddenLayerSize);
net.trainParam.mc = 0.97;
net.trainParam.lr = 0.87; % a large learing rate (lr) could accelerate the learning speed for convergence
net.trainParam.lr_inc = 1.03;
net.trainParam.lr_dec = 0.81;
net.trainParam.goal = 1e-6;
net.trainParam.min_grad = 1e-5;
net.trainParam.max_fail = 20;
net.trainParam.epoch = 1000;

% Partition the dataset for training, validation, and testing purposes
net.divideFcn = 'dividerand';
net.divideParam.trainRatio = 0.70;
net.divideParam.valRatio = 0.20;
net.divideParam.testRatio = 0.10;

% Set up the real-time display for overseeing learning process
net.performFcn = 'mse';
net.adaptFcn = 'none';
net.trainParam.showCommandLine = true;
net.trainParam.showWindow = true;

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

% % Plot the training performance in the output of triple products between the observed and predicted
% trainPerformance = nneval(net, trainInp, trainTarg);
% figure
% index = 1 : length(trainInd);
% subplot(3, 1, 1)
% plot(index, trainTarg(1, :), 'r-o', index, trainOut(1, :), 'b-.s'); 
% legend({'Observed char', 'Predicted char'}, 'Location','northwest'); xlabel('Index of training samples'); ylabel('Output');
% title(['Whole training perforance in MSE = ' num2str(trainPerformance)]);
% subplot(3, 1, 2)
% plot(index, trainTarg(2, :), 'r-o', index, trainOut(2, :), 'b-.s'); 
% legend({'Observed tar', 'Predicted tar'}, 'Location','northwest'); xlabel('Index of training samples'); ylabel('Output');
% subplot(3, 1, 3)
% plot(index, trainTarg(3, :), 'r-o', index, trainOut(3, :), 'b-.s'); 
% legend({'Observed gas', 'Predicted gas'}, 'Location','northwest'); xlabel('Index of training samples'); ylabel('Output');
% 
% % Plot the validation performance in the output of triple products between the observed and predicted
% if validation
%     valPerformance = nneval(net, valInp, valTarg);
%     figure
%     index = 1 : length(valInd);
%     subplot(3, 1, 1)
%     plot(index, valTarg(1, :), 'r-o', index, valOut(1, :), 'b-.s'); 
%     legend({'Observed char', 'Predicted char'}, 'Location','northwest'); xlabel('Index of training samples'); ylabel('Output');
%     title(['Whole validation perforance in MSE = ' num2str(valPerformance)]);
%     subplot(3, 1, 2)
%     plot(index, valTarg(2, :), 'r-o', index, valOut(2, :), 'b-.s'); 
%     legend({'Observed tar', 'Predicted tar'}, 'Location','northwest'); xlabel('Index of training samples'); ylabel('Output');
%     subplot(3, 1, 3)
%     plot(index, valTarg(3, :), 'r-o', index, valOut(3, :), 'b-.s'); 
%     legend({'Observed gas', 'Predicted gas'}, 'Location','northwest'); xlabel('Index of training samples'); ylabel('Output');
% end
% 
% % Plot the testing performance in the output of triple products between the observed and predicted
% if testing
%     testPerformance = nneval(net, testInp, testTarg);
%     figure
%     index = 1 : length(testInd);
%     subplot(3, 1, 1)
%     plot(index, testTarg(1, :), 'r-o', index, testOut(1, :), 'b-.s'); 
%     legend({'Observed char', 'Predicted char'}, 'Location','northwest'); xlabel('Index of training samples'); ylabel('Output');
%     title(['Whole testing perforance in MSE = ' num2str(testPerformance)]);
%     subplot(3, 1, 2)
%     plot(index, testTarg(2, :), 'r-o', index, testOut(2, :), 'b-.s'); 
%     legend({'Observed tar', 'Predicted tar'}, 'Location','northwest'); xlabel('Index of training samples'); ylabel('Output');
%     subplot(3, 1, 3)
%     plot(index, testTarg(3, :), 'r-o', index, testOut(3, :), 'b-.s'); 
%     legend({'Observed gas', 'Predicted gas'}, 'Location','northwest'); xlabel('Index of training samples'); ylabel('Output');
% end

 % Plot the regression performance of training, validation, testing, and the overall bteween the obserables and the predicted ones
 figure
 % For overall case
 subplot(2, 2, 1)
 Overall_obs = [trainTarg valTarg testTarg];
 Overall_pred = [trainOut valOut testOut];
 plot(Overall_obs, Overall_pred,'ro','MarkerSize', 4); hold on
 maximum = max(Overall_obs(:)); minimum = min(Overall_obs(:));
 x = linspace(minimum, maximum, length(Overall_obs));
 plot(x, x, 'b');
 Rsq_overall = 1 - sum((Overall_obs - Overall_pred).^2)/sum((Overall_obs - mean(Overall_obs)).^2);
 title(['Overall performance R^2 = ' num2str(Rsq_overall)]); xlabel('Observed values'); ylabel('Predicted values');

 % For training case
 subplot(2, 2, 2)
 plot(trainTarg, trainOut,'ro','MarkerSize', 4); hold on
 maximum = max(trainTarg(:)); minimum = min(trainTarg(:));
 x = linspace(minimum, maximum, length(trainTarg));
 plot(x, x, 'b');
 Rsq_training = 1 - sum((trainTarg - trainOut).^2)/sum((trainTarg - mean(trainTarg)).^2);
 title(['Training performance R^2 = ' num2str(Rsq_training)]); xlabel('Observed values'); ylabel('Predicted values');

% For validation case
 subplot(2, 2, 3)
 plot(valTarg, valOut,'ro','MarkerSize', 4); hold on
 maximum = max(valTarg(:)); minimum = min(valTarg(:));
 x = linspace(minimum, maximum, length(valTarg));
 plot(x, x, 'b');
 Rsq_validation = 1 - sum((valTarg - valOut).^2)/sum((valTarg - mean(valTarg)).^2);
 title(['Validation performance R^2 = ' num2str(Rsq_validation)]); xlabel('Observed values'); ylabel('Predicted values');
 
 % For testing case
 subplot(2, 2, 4)
 plot(testTarg, testOut,'ro','MarkerSize', 4); hold on
 maximum = max(testTarg(:)); minimum = min(testTarg(:));
 x = linspace(minimum, maximum, length(testTarg));
 plot(x, x, 'b');
 Rsq_testing = 1 - sum((testTarg - testOut).^2)/sum((testTarg - mean(testTarg)).^2);
 title(['Testing performance R^2 = ' num2str(Rsq_testing)]); xlabel('Observed values'); ylabel('Predicted values');

 % Visualize weights between Input, Hidden, and Output layers
 % numHiddenLayer = length(hiddenLayerSize);
 % % figure
 % for i = 1 : numHiddenLayer + 1
 %      if i == 1
 %          figure(i + 6) % prevent the overwriting of the previously plotted Figures by increasing the indice of the new figure
 %          glyphplot(net.weight{1, i}); title('Weights from Input Layer to Hidden Layer 1');
 %      end
 %      if i > 1
 %          figure(i + 6)
 %          glyphplot(net.weight{1, i}); title(['Weights from Hidden Layer ' num2str(i-1) ' to Hidden Layer ' num2str(i)]);
 %      end
 %       if i == numHiddenLayer + 1
 %          figure(i + 6)
 %          glyphplot(net.weight{1, i}); title(['Weights from Hidden Layer ' num2str(i-1) ' to Output Layer']);
 %       end
 % end