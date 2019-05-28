function [tau] = estimateAutocorrelationTime(cov_s)
%ESTIMATEAUTOCORRELATIONTIME Estimates the autocorrelation time as:
% 1 + 2 * sum autocorr_i
%   @param cov_s: covariance matrix
%   @return: autocorrelation time [n / sampling rate]
%
% Author: Agustin PICARD, intern @ Scalian with Valentin RESSEGUIER as
% supervisor
%
    
tau = 1 + 2 * sum(estimateAutocorrelation(cov_s));

end

