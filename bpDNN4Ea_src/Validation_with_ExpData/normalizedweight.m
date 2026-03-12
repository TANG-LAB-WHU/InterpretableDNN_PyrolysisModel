function w = normalizedweight(w_0,alpha)
% w_0: the final value of sample weight at the end of the pyrolysis temperature domain
%  alpha: the dgree of conversion

w = (1 - (1 - w_0)*alpha) * 100;

end