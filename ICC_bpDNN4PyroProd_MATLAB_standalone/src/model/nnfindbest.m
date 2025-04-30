function [best, pos]= nnfindbest(model, option, trainPerf, valPerf, testPerf)
%NNFINDBEST finds best model from all models.
%   [best, bestloss, bestpos] = FINDBEST(model, option, trainerr, ...
%           valerr, testerr) Return a model best performance. 

[~, pos] = min(trainPerf);
if option.validation && ~all(isnan(valPerf))
    % Only use validation performance if it has valid values
    [~, val_pos] = min(valPerf(~isnan(valPerf)));
    if ~isempty(val_pos)
        pos = val_pos;
    end
end
if option.testing && ~all(isnan(testPerf))
    % Only use test performance if it has valid values
    [~, test_pos] = min(testPerf(~isnan(testPerf)));
    if ~isempty(test_pos)
        pos = test_pos;
    end
end
best = model{pos};