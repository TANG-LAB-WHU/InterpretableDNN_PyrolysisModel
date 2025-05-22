function [net, tr] = trainWithEarlyStopping(net, X, Y, threshold)
% TRAINWITHEARLYSTOPPING Train neural network with early stopping based on MSE threshold
% Input parameters:
%   net: The neural network to train
%   X: Input data
%   Y: Target data
%   threshold: MSE threshold for early stopping
%
% Output parameters:
%   net: Trained neural network
%   tr: Training record

% Initialize training parameters
max_epochs = net.trainParam.epoch;
early_stop = false;
epoch_count = 0;
tr = struct();
tr.epoch = [];
tr.perf = [];
tr.vperf = [];
tr.tperf = [];
tr.goal = net.trainParam.goal;
tr.stop = 'max_epochs';
tr.gradient = [];
tr.val_fail = 0;

% Get data division
[trainInd, valInd, testInd] = divideind(size(X, 2), 1:size(X, 2), [], []);
if ~isempty(net.divideFcn)
    [trainInd, valInd, testInd] = feval(net.divideFcn, size(X, 2), ...
        net.divideParam.trainRatio, ...
        net.divideParam.valRatio, ...
        net.divideParam.testRatio);
end

tr.trainInd = trainInd;
tr.valInd = valInd;
tr.testInd = testInd;

% Training loop
while ~early_stop && epoch_count < max_epochs
    epoch_count = epoch_count + 1;
    
    % Train for one epoch
    [net, tr_epoch] = nntrain(net, X, Y, 1);
    
    % Record training metrics
    tr.epoch(end+1) = epoch_count;
    tr.perf(end+1) = tr_epoch.perf(end);
    if ~isempty(tr_epoch.vperf)
        tr.vperf(end+1) = tr_epoch.vperf(end);
    end
    if ~isempty(tr_epoch.tperf)
        tr.tperf(end+1) = tr_epoch.tperf(end);
    end
    tr.gradient(end+1) = tr_epoch.gradient(end);
    
    % Check if validation performance is above threshold
    if ~isempty(tr.vperf) && tr.vperf(end) > threshold
        val_fail_count = tr.val_fail + 1;
        tr.val_fail = val_fail_count;
        
        % If validation error is still high after 5 epochs, stop early
        if val_fail_count >= 5 && epoch_count >= 10
            early_stop = true;
            tr.stop = 'early_stopping_threshold';
            fprintf('Early stopping at epoch %d: validation MSE (%.6f) above threshold (%.6f)\n', ...
                epoch_count, tr.vperf(end), threshold);
        end
    else
        tr.val_fail = 0;
    end
    
    % Check other stopping criteria
    if ~isempty(tr_epoch.stop) && ~strcmp(tr_epoch.stop, 'max_epochs')
        early_stop = true;
        tr.stop = tr_epoch.stop;
    end
    
    % Display progress every 10 epochs
    if mod(epoch_count, 10) == 0
        fprintf('Epoch %d/%d: MSE = %.6f', epoch_count, max_epochs, tr.perf(end));
        if ~isempty(tr.vperf)
            fprintf(', Val MSE = %.6f', tr.vperf(end));
        end
        if ~isempty(tr.tperf)
            fprintf(', Test MSE = %.6f', tr.tperf(end));
        end
        fprintf('\n');
    end
end

end
