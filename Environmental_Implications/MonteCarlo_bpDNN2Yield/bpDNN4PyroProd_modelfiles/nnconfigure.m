function [net, data] = nnconfigure(net, P, T)
%NNCONFIGURE configures paramaters of networks (i.e. net.numInputs,
%       net.numOutputs, and net.weights).

if ~isequal(size(P, 2), size(T, 2))
    error('NN:Preprocess:misMatch','the size of Inputs never matched with Targets.');
end

net.numInput = size(P, 1);
net.numOutput = size(T, 1);
net.layer{end}.size = net.numOutput;

% Initialize weights and biases.
net = nninit(net);

% Normalize inputs and targets.
switch lower(net.processFcn)
    case {'mapminmax', 'minmax'}
        [pn, ps] = mapminmax(P, net.processParam);
        [tn, ts] = mapminmax(T, net.processParam);
    case {'mapstd', 'std'}
        [pn, ps] = mapstd(P, net.processParam);
        [tn, ts] = mapstd(T, net.processParam);
end

% data.p = P;
% data.t = T;
data.P = pn;
data.T = tn;
data.PS = ps;
data.TS = ts;

N = size(P, 2);

% Rertieve index of dataset for training, validation and testing after data splitting
[trainInd, valInd, testInd] = feval(net.divideFcn, N, net.divideParam);
data.trainInd = trainInd;
data.valInd = valInd;
data.testInd = testInd;

end