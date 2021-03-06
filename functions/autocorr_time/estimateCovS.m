function [autocorrelation] = estimateCovS(cov_s)
%ESTIMATEAUTOCORRELATION Estimates the autocorrelation function given a
%covariance matrix
%   @param cov_s: covariance matrix
%   @return: autocorrelation function with as many lags as cov_s's columns
%
% Author: Agustin PICARD, intern @ Scalian with Valentin RESSEGUIER as
% supervisor
%

if isnumeric(cov_s)
    N = size(cov_s, 2);
    autocorrelation = zeros(N, 1);
    
    for i = 1 : N
        c = 0;
        for j = 1 : N - i + 1
            c = c + cov_s(j,j+i-1);
        end
        autocorrelation(i) = c * N / (N - i + 1);
    end
    
    
elseif iscell(cov_s)
    N = length(cov_s);
    N_i = length(cov_s{2});
    autocorrelation = zeros(N , 1);
    
    % equivalent of map(lambda x: sum(x), cov_s)
    diagSums = cellfun(@sum, cov_s);
    for i = 1 : N
        autocorrelation(i) = diagSums(i) * N / (N_i - i + 2);
    end
    
end


end

