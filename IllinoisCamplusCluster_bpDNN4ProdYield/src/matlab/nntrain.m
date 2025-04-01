function [net, tr] = nntrain(net, x, y)
% NNTRAIN Train the neural network
% From input and output (target) values

    % Parse hyperparamters
    if ~isfield(net.trainParam, 'showWindow'), net.trainParam.showWindow = true; end
    if ~isfield(net.trainParam, 'showCommandLine'), net.trainParam.showCommandLine = false; end
    if ~isfield(net.trainParam, 'goal'), net.trainParam.goal = 1e-6; end
    if ~isfield(net.trainParam, 'epoch'), net.trainParam.epoch = 100; end
    if ~isfield(net.trainParam, 'max_fail'), net.trainParam.max_fail = 10; end
    if ~isfield(net.trainParam, 'min_grad'), net.trainParam.min_grad = 1e-8; end
    if ~isfield(net.trainParam, 'lr'), net.trainParam.lr = 0.01; end
    if ~isfield(net.trainParam, 'mc'), net.trainParam.mc = 0.9; end
    if ~isfield(net.trainParam, 'lr_inc'), net.trainParam.lr_inc = 1.05; end
    if ~isfield(net.trainParam, 'lr_dec'), net.trainParam.lr_dec = 0.7; end
    
    % Preprocess target and input
    [x, net] = nnpreprocess(net, x);
    y = nnprepare(net, y);
    
    [tInd, vInd, tuInd] = nnconfigure(net, size(y, 2));
    
    net = nninit(net);
    
    tX = x(:, tInd);
    tY = y(:, tInd);
    vX = x(:, vInd);
    vY = y(:, vInd);
    tuX = x(:, tuInd);
    tuY = y(:, tuInd);
    
    if net.trainParam.showWindow, nnupdatefigure(-1, net, [], [], [], [], [], [], [], []); end

    [best_vperf, tr] = nnfindbest(net, 1, [], [], [], [], []);
    
    if ~isempty(best_vperf)
        best_tr = tr;
        best_net = net;
        best_epoch = 0;
        best_perf = tr.perf;
        best_test_perf = tr.test_perf;
        best_vperf = tr.vperf;
    end
    
    % Save original learning rate for adaptive adjustments
    original_lr = net.trainParam.lr;
    prev_perf = Inf;  % To track performance improvement
    
    tr.epoch = 0;
    tr.perf = 0;
    tr.vperf = 0;
    tr.tperf = 0;
    
    failCount = 0;
    
    for epoch = 1:net.trainParam.epoch
        tr.epoch = epoch;
        
        [net, tr, perf] = nnbp(net, tX, tY);
        tr.perf = perf;
        
        % Calculate validation and test performance
        if ~isempty(vInd)
            vOut = nnff(net, vX);
            tr.vperf = nneval(net, vX, vY);
        else
            tr.vperf = [];
        end
        
        if ~isempty(tuInd)
            tuOut = nnff(net, tuX);
            tr.tperf = nneval(net, tuX, tuY);
        else
            tr.tperf = [];
        end

        % ********** Adaptive learning rate adjustment code **********
        % Use lr_inc and lr_dec parameters to dynamically adjust learning rate
        if epoch > 1
            % If performance improves, increase learning rate
            if perf < prev_perf
                net.trainParam.lr = net.trainParam.lr * net.trainParam.lr_inc;
                if net.trainParam.showCommandLine
                    fprintf('Performance improved: increasing learning rate to %f\n', net.trainParam.lr);
                end
            % If performance worsens, decrease learning rate
            else
                net.trainParam.lr = net.trainParam.lr / net.trainParam.lr_dec;
                if net.trainParam.showCommandLine
                    fprintf('Performance worsened: decreasing learning rate to %f\n', net.trainParam.lr);
                end
            end
            
            % Limit learning rate to reasonable range
            net.trainParam.lr = min(net.trainParam.lr, 10 * original_lr);
            net.trainParam.lr = max(net.trainParam.lr, 0.001 * original_lr);
        end
        prev_perf = perf;
        % ********** End of adaptive learning rate adjustment code **********
        
        % Update figure
        if net.trainParam.showWindow, nnupdatefigure(epoch, net, tr, tInd, vInd, tuInd, tY, vY, tuY, [perf, tr.vperf, tr.tperf]); end
        
        % Display output
        if net.trainParam.showCommandLine
            if ~isempty(tr.vperf)
                fprintf('Epoch: %4d, Training MSE: %10.8f, Validation MSE: %10.8f\n', tr.epoch, tr.perf, tr.vperf);
            else
                fprintf('Epoch: %4d, Training MSE: %10.8f\n', tr.epoch, tr.perf);
            end
        end
        
        % Check if best validation MSE
        if ~isempty(tr.vperf) && (tr.vperf < best_vperf || epoch == 1)
            best_tr = tr;
            best_net = net;
            best_epoch = epoch;
            best_perf = tr.perf;
            best_vperf = tr.vperf;
            failCount = 0;
        else
            failCount = failCount + 1;
        end
        
        % Stop criteria check
        if nnstopcriteria(net, tr, best_tr, failCount, epoch)
            break;
        end
    end
    
    tr.best_epoch = best_epoch;
    tr.best_perf = best_perf;
    tr.best_vperf = best_vperf;
    tr.best_tperf = best_test_perf;
    tr.trainInd = tInd;
    tr.valInd = vInd;
    tr.testInd = tuInd;
    
    % Use best network
    net = best_net;
end

function perf = evaluate_network(net, input, target)
% Evaluate the performance of networks when normalized input and target is fed into .
%   inputs:
%       net - the trained neural network.
%       target - normalized input.
%       output - normalized target.
%   outputs:
%       perf - network performance.

global TS

outLayer = nnff(net, input, target);
output = outLayer{end};

% Reversed normalization for target and output.
target = nnpostprocess(net.processFcn, target, TS);
output = nnpostprocess(net.processFcn, output, TS);

performFcn = net.performFcn;
switch lower(performFcn)
    case {'mse'}
        perf = mse(target - output);
    case {'mae'}
        perf = mae(target - output);
    case {'sae'}
        perf = sae(target - output);
    case {'sse'}
        perf = sse(target - output);
end