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
% quantile = sorted_data(:, :, quantile(1));

len_sz = length(size(sorted_data));
sorted_data = permute(sorted_data,[1:(dim-1) (dim+1):len_sz dim] );
siz = size(sorted_data);
sorted_data = reshape(sorted_data,[prod(siz(1:end-1)) siz(end)] );
quantile = sorted_data(:, quantile(1));
quantile = reshape(quantile,[siz(1:end-1) 1] );
quantile = permute(quantile,[1:(dim-1) len_sz (dim):(len_sz-1)] );

end

