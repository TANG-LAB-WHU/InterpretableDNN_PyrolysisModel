function [net, tr] = nntrain(net, P, T)
%NNTRAIN trains a network with P inputs and T targets according to the
%       training strategy which are defined in net.
%   Inputs:
%           net - a feedforward neural network created by NNCREATE.
%           P - Inputs of networks. Inputs could be the measurement of
%           samples. Each column is a sample and each row is a feature. The
%           number of colomns are the number of samples.
%           T - Targets of outputs. Each column is the target of a samples
%           and each row is an output. The number of colomons is equal to
%           the number of samples. The number of rows is equal to outputs.
%   Outputs:
%           net - A trained network.
%           tr - Training record containing performance information.
%   Example:
%           [net, tr] = NNTRAIN(net, P, T)

% Data configuring.
[net, data] = nnconfigure(net, P, T);

% Data preparation.
[net, tr, data, option] = nnprepare(net, data);

% Make sure tr is initialized
if ~exist('tr', 'var') || isempty(tr)
    tr = struct();
    tr.trainInd = [];
    tr.valInd = [];
    tr.testInd = [];
    tr.perf = [];
    tr.vperf = [];
    tr.tperf = [];
    tr.best_perf = [];
    tr.best_vperf = [];
    tr.best_tperf = [];
    tr.epoch = [];
    tr.time = [];
    tr.gradient = [];  
    tr.stop = [];
    tr.best_epoch = [];
    tr.num_epochs = [];
    tr.val_fail = [];
end

t = cputime;

% Initialize error.
error = [];

N = size(data.P, 2);

% Assign input and outputs.
trainInd = data.trainInd;
valInd = data.valInd;
testInd = data.testInd;

Ptrain = data.P(:, trainInd);
Ttrain = data.T(:, trainInd);

validation = option.validation;
testing = option.testing;
Pval = []; Tval = [];
Ptest = []; Ttest = [];

if validation
    Pval = data.P(:, valInd);
    Tval = data.T(:, valInd);
end

if testing
    Ptest = data.P(:, testInd);
    Ttest = data.T(:, testInd);
end

epoch = 0;
val_fail = 0;

pdW = data.pdW;
pdb = data.pdb;

fig = data.figure;
fail = 0;
best_vperf = inf;
best_perf = inf;
best_tperf = inf;

best_net = net;

best_epoch = 0;

max_fail = net.trainParam.max_fail;
goal = net.trainParam.goal;
epochs = net.trainParam.epoch;
min_grad = net.trainParam.min_grad;

while 1
    epoch = epoch + 1;
    
    % Search gradient of network.
    [Atrain, Otrain] = nnff(net, Ptrain);
    error = Ttrain - Otrain;
    
    % Backpropagation (BP) to calibrate weights and biases of the network.
    [net, gradient, dW, db] = nnbp(net, Ptrain, Ttrain, error, Atrain, pdW, pdb);
    
    Etrain = evaluate_network(net.performFcn, error);
    tr.perf = [tr.perf, Etrain];
    tr.gradient = [tr.gradient, gradient];
    tr.trainInd = [tr.trainInd, {trainInd}];
    tr.valInd = [tr.valInd, {valInd}];
    tr.testInd = [tr.testInd, {testInd}];
    tr.epoch = [tr.epoch, epoch];
    tr.time = [tr.time, cputime - t];
    
    if gradient <= min_grad
        tr.stop = [tr.stop, {'Minimum gradient reached.'}];
        break
    end
    
    % Validation performance.
    if validation
        [Aval, Oval] = nnff(net, Pval);
        Eval = evaluate_network(net.performFcn, Tval - Oval);
        tr.vperf = [tr.vperf, Eval];
    end
    
    % Test performance.
    if testing
        [Atest, Otest] = nnff(net, Ptest);
        Etest = evaluate_network(net.performFcn, Ttest - Otest);
        tr.tperf = [tr.tperf, Etest];
    end
    
    % Update tain and validation errors for plotting.
    if net.trainParam.showWindow
        % update train and val errors for plotting
        h = figure(fig);
        if ~ishandle(h)
            fprintf('User closed the GUI window. Training process is interrupted.\n\n');
            tr.stop = [tr.stop, {'User closed the GUI window.'}];
            break
        end
%         subplot(3,1,1); plot(tr.epoch, tr.perf); title('Trainging Error'); xlabel('Epoch'); ylabel('Error');
%         subplot(3,1,2); plot(tr.epoch, tr.gradient); title('Gradient'); xlabel('Epoch'); ylabel('Gradient');
%         subplot(3,1,3); 
        
        set(0, 'CurrentFigure', fig);
        if validation && testing
            subplot(3,1,1); plot(tr.epoch, tr.perf, 'r-', tr.epoch, tr.vperf, 'b:', tr.epoch, tr.tperf, 'g-.'); 
            title('Trainging Error'); xlabel('Epoch'); ylabel('Error');
            legend({'Train', 'Validation', 'Test'}, 'Location', 'northeast');
        elseif validation
            subplot(3,1,1); plot(tr.epoch, tr.perf, 'r-', tr.epoch, tr.vperf, 'b:'); 
            title('Trainging Error'); xlabel('Epoch'); ylabel('Error');
            legend({'Train', 'Validation'}, 'Location', 'northeast');
        elseif testing
            subplot(3,1,1); plot(tr.epoch, tr.perf, 'r-', tr.epoch, tr.tperf, 'g-.'); 
            title('Trainging Error'); xlabel('Epoch'); ylabel('Error');
            legend({'Train', 'Test'}, 'Location', 'northeast');
        else
            subplot(3,1,1); plot(tr.epoch, tr.perf, 'r-'); 
            title('Trainging Error'); xlabel('Epoch'); ylabel('Error');
            legend({'Train'}, 'Location', 'northeast');
        end
        subplot(3,1,2); plot(tr.epoch, tr.gradient); title('Gradient'); xlabel('Epoch'); ylabel('Gradient');
        subplot(3,1,3); 
        if validation
            plot(tr.epoch(1:end-1), diff(tr.vperf)); title('\Delta Error'); xlabel('Epoch'); ylabel('Error diff.');
        else
            plot(tr.epoch(1:end-1), diff(tr.perf)); title('\Delta Error'); xlabel('Epoch'); ylabel('Error diff.');
        end
        drawnow;
    end
    
    % Re-assign pdW, pdb; (crucial)
    pdW = dW;
    pdb = db;
    
    % Update best train performance.
    if Etrain < best_perf
        best_perf = Etrain;
        tr.best_perf = [tr.best_perf, best_perf];
    else
        tr.best_perf = [tr.best_perf, best_perf];
    end
    
    % Update best validation performance.
    if validation
        if Eval < best_vperf
            best_vperf = Eval;
            best_net = net;
            best_epoch = epoch;
            tr.best_vperf = [tr.best_vperf, best_vperf];
            tr.best_epoch = [tr.best_epoch, best_epoch];
            val_fail = 0;
        else
            tr.best_vperf = [tr.best_vperf, best_vperf];
            tr.best_epoch = [tr.best_epoch, best_epoch];
            val_fail = val_fail + 1;
        end
    end
    
    % Update best test performance.
    if testing
        if Etest < best_tperf
            best_tperf = Etest;
            tr.best_tperf = [tr.best_tperf, best_tperf];
        else
            tr.best_tperf = [tr.best_tperf, best_tperf];
        end
    end
    
    tr.val_fail = [tr.val_fail, val_fail];
    
    if val_fail >= max_fail
        tr.stop = [tr.stop, {'Maximum validation failures.'}];
        break
    end
    
    if best_perf <= goal
        tr.stop = [tr.stop, {'Performance goal met.'}];
        break
    end
    
    if epoch >= epochs
        tr.stop = [tr.stop, {'Maximum epoch reached.'}];
        break
    end
    
    % user cancel
    if ~isempty(tr.stop)
        break
    end
    
    fprintf('Epoch: %d/%d (%.1f%%), TrainMSE: %.6f', epoch, epochs, epoch/epochs*100, Etrain);
    if validation
        fprintf(', ValMSE: %.6f', Eval);
    end
    if testing
        fprintf(', TestMSE: %.6f', Etest);
    end
    fprintf(', Grad: %.6f\n', gradient);
end

tr.num_epochs = epoch;
if validation
    net = best_net;
end

end

function perf = evaluate_network(performFcn, error)
% Helper function to evaluate performance based on error
switch lower(performFcn)
    case 'mse'
        perf = mean(mean(error.^2));
    case 'sse'
        perf = sum(sum(error.^2));
    case 'mae'
        perf = mean(mean(abs(error)));
    case 'sae'
        perf = sum(sum(abs(error)));
    otherwise
        error('NN:Performance:unknown', 'Unknown performance function: %s', performFcn);
end
end