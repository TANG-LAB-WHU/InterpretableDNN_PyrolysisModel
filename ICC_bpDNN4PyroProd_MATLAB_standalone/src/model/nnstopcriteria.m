function [stop, fail] = nnstopcriteria(net, i, gradient, trainPerf, validPerf)
% NNSTOPCRITERIA triggers a stopping criterion and then returns stop category.

persistent count; % count for validation error violation.
if isempty(count)
    count = 0;
end

stop = 'normal';
if ~isempty(validPerf) && i > 1 && ~isnan(validPerf(i)) && ~isnan(validPerf(i-1))
    if validPerf(i) - validPerf(i - 1) > 0
        count = count + 1;
    else
        count = 0;
    end
end

fail = count;

if  i == net.trainParam.epoch
    stop = 'maxiteration';
    return;
end

if  ~isnan(trainPerf(i)) && trainPerf(i) < net.trainParam.goal
    stop = 'goal';
    return;
end

if ~isnan(gradient(i)) && gradient(i) < net.trainParam.min_grad
    stop = 'mingrad';
    return;
end

if  fail == net.trainParam.max_fail
    stop = 'maxfail';
    return;
end