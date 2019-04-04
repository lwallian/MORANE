function [tau] = estimateAutocorrelationTime(cov_s)
%ESTIMATEAUTOCORRELATIONTIME Estimates the autocorrelation time as:
% 1 + 2 * sum autocorr_i
%   Use the fact that data types are accesible at runtime to make it useful
%   for both cells and matrices
%   @param cov_s: covariance matrix
%
% Author: Agustin PICARD, intern @ Scalian with Valentin RESSEGUIER as
% supervisor
%
    
tau = 1 + 2 * sum(estimateAutocorrelation(cov_s));

end

