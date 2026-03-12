function T = nnpostprocess(normalizationFcn, TN, TS)
% NNPOSTPROCESS reverses outputs of the network.
%   T = POSTPROCESS(net, TN, TS) reverse outputs of the network 
%       according to target normalization setting (TS);

switch lower(normalizationFcn)
    case {'mapminmax', 'minmax'}
        T = mapminmax('reverse', TN, TS);
    case {'mapstd', 'std'}
        T = mapstd('reverse', TN, TS);
end