function [tau] = correlationTimeLMS(covMatrix, chronos)
%CORRELATIONTIMELMS Estimates the correlation time through the typical
%estimator but on an lms filtered correlation function so as to filter out
%all the periodicity.
%   @param covMatrix: covariance matrix of the velocity field
%   @param chronos: set of chronos of the POD
%   @return tau: the estimated correlation time
%
% Author: Agustin PICARD, intern @ Scalian with Valentin RESSEGUIER as
% supervisor
%

cov_s = smallScaleVelocityCov(covMatrix, chronos);
correlation = estimateAutocorrelation(cov_s);
period = periodicityFromAutocorrelation(cov_s);
[filteredCorrelation, ~] = minimalVarianceLMS(correlation, 500, period);

if length(filteredCorrelation) < 5 * period
    tau = 1 + 2 * sum(filteredCorrelation);
else
    tau = 1 + 2 * sum(filteredCorrelation(1 : 5 * period));
end

end

