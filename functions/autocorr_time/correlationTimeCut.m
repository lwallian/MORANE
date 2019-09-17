function [tau] = correlationTimeCut(covMatrix, chronos)
%CORRELATIONTIMECUT Estimates the correlation time assuming the transcient
%finishes at around 5 Strouhal periods
%   @param covMatrix: covariance matrix of the velocity field
%   @param chronos: set of chronos of the POD
%   @return tau: the estimated correlation time
%
% Author: Agustin PICARD, intern @ Scalian with Valentin RESSEGUIER as
% supervisor
%

% Estimate the autocorrelation and its periodicity
cov_s = smallScaleVelocityCov(covMatrix, chronos);
correlation = estimateAutocorrelation(cov_s);
period = periodicityFromAutocorrelation(covMatrix);

% Apply the estimator
tau = 1 + 2 * sum(correlation(1 : 5 * period));

end

