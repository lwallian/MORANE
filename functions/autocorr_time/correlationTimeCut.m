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

cov_s = smallScaleVelocityCov(covMatrix, chronos);
correlation = estimateAutocorrelation(cov_s);
period = periodicityFromAutocorrelation(covMatrix);

tau = 1 + 2 * sum(correlation(1 : 5 * period));

end

