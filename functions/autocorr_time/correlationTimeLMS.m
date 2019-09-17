function [tau] = correlationTimeLMS(covMatrix, chronos, dt)
%CORRELATIONTIMELMS Estimates the correlation time through the typical
%estimator but on an lms filtered correlation function so as to filter out
%all the periodicity.
%   @param covMatrix: covariance matrix of the velocity field
%   @param chronos: set of chronos of the POD
%   @param dt: time step of the DNS simulation
%   @return tau: the estimated correlation time
%
% Author: Agustin PICARD, intern @ Scalian with Valentin RESSEGUIER as
% supervisor
%

% Estimate the correlation function and its periodicity
cov_s = smallScaleVelocityCov(covMatrix, chronos);
correlation = estimateAutocorrelation(cov_s);
period = periodicityFromAutocorrelation(covMatrix);

% Apply the LMS filter with a fixed period and a maximum number of taps set
% to 500
[filteredCorrelation, ~] = minimalVarianceLMS(correlation, 500, period);

% If the correlation is not large enough, the error introduced by the
% variance should stay low so we just apply the filter to the whole thing
if length(filteredCorrelation) < 5 * period
    tau = 1 + 2 * sum(filteredCorrelation);
else
    tau = 1 + 2 * sum(filteredCorrelation(1 : 5 * period));
end

% If the estimation doesn't seem right, just use the heterogeneous
% estimator
if tau < 1
    tau = htgenCorrelationTime(covMatrix, chronos, dt);
end

end

