function [tau] = simpleCorrelationTime(covMatrix, chronos, dt)
%SIMPLECORRELATIONTIME Estimates the correlation time of a given POD
%through a simple estimator : sqrt(2 * mean(cov_s^2) / mean((dcov_s / dt)^2))
%   @param covMatrix: covariance matrix of the velocity field
%   @param chronos: set of chronos of the POD
%   @return tau: the estimated correlation time
%
% Author: Agustin PICARD, intern @ Scalian with Valentin RESSEGUIER as
% supervisor
%

cov_s = smallScaleVelocityCov(covMatrix, chronos);
correlation = estimateAutocorrelation(cov_s);
tau = sqrt(2 * mean((correlation).^2) / mean((diff(correlation) ./ dt).^2));

end

