function [tau] = htgenCorrelationTime2(covMatrix, chronos, dt)
%SIMPLECORRELATIONTIME Estimates the correlation time of a given POD
%through a simple estimator : sqrt(2 * mean(cov_s^2) / mean((dcov_s / dt)^2))
%   @param covMatrix: covariance matrix of the velocity field
%   @param chronos: set of chronos of the POD
%   @return tau: the estimated correlation time
%
% Author: Agustin PICARD, intern @ Scalian with Valentin RESSEGUIER as
% supervisor
%

c_prime = smallScaleVelocityCov(covMatrix, chronos);
correlation = estimateCovS(c_prime);
dt_correlation = diff(correlation);
dt_correlation = [ 0; dt_correlation ]; 
% The time derivative is always 0 at delta t = 0 because of the symetry
tau = sqrt(2 * mean(correlation.^2) / mean( dt_correlation.^2));

end

