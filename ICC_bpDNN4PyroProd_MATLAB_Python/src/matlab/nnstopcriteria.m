function stop = nnstopcriteria(net, tr, best_tr, failCount, epoch)
% NNSTOPCRITERIA Check if any stopping criteria for neural network training are met
% Returns a boolean indicating whether training should stop

    stop = false;
    
    % Check if maximum number of epochs reached
    if epoch >= net.trainParam.epoch
        if net.trainParam.showCommandLine
            fprintf('Maximum number of epochs (%d) reached, stopping training.\n', net.trainParam.epoch);
        end
        stop = true;
        return;
    end
    
    % Check if performance goal reached
    if tr.perf <= net.trainParam.goal
        if net.trainParam.showCommandLine
            fprintf('Performance goal (%g) reached, stopping training.\n', net.trainParam.goal);
        end
        stop = true;
        return;
    end
    
    % Check if minimum gradient reached
    if epoch > 1 && abs(tr.perf - best_tr.perf) < net.trainParam.min_grad
        if net.trainParam.showCommandLine
            fprintf('Minimum gradient (%g) reached, stopping training.\n', net.trainParam.min_grad);
        end
        stop = true;
        return;
    end
    
    % Check if maximum validation failures reached
    if failCount >= net.trainParam.max_fail
        if net.trainParam.showCommandLine
            fprintf('Maximum validation failures (%d) reached, stopping training.\n', net.trainParam.max_fail);
        end
        stop = true;
        return;
    end
end