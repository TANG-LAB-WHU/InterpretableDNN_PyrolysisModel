function PN = nnpreprocess(normalizationFcn, P, PS)
% NNPREPROCESS returns normalized matrix of a given matrix. Atrributes were
% ordered in the matrix by column

switch lower(normalizationFcn)
    case {'mapminmax', 'minmax'}
        PN = mapminmax('apply', P, PS);
    case {'mapstd', 'std'}
        PN = mapstd('apply', P, PS);
end