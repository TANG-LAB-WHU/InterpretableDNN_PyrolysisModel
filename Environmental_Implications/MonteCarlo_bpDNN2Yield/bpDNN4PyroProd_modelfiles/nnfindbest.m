function [best, pos]= nnfindbest(model, option, trainPerf, valPerf, testPerf)
%NNFINDBEST finds best model from all models.
%   [best, bestloss, bestpos] = FINDBEST(model, option, trainerr, ...
%           valerr, testerr) Return a model best performance. 

[~, pos] = min(trainPerf);
if option.validation
    [~, pos] = min(valPerf);
end
if option.testing
    [~, pos] = min(testPerf);
end
best = model{pos};