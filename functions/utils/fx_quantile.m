function [quantile] = fx_quantile(data, p, dim)
%FX_QUANTILE Estimates the empirical quantile along a certain dimension of the
%provided dataset
%   @param data: dataset
%   @param quantile: the probability we want to achieve
%   @param dim: dimension along which the quantile will be estimated
%   @return quantile: estimated quantile
%
% Author: Agustin PICARD, intern @ Scalian with Valentin RESSEGUIER as
% supervisor
%

sorted_data = sort(data, dim, 'descend');
empirical_cdf = (1 : size(data, dim)) ./ size(data, dim);

quantile = find(empirical_cdf >= p);
% Hardcoded dimensions for the use case in this code!
quantile = sorted_data(:, :, quantile(1));

end

