function [sweep] = aperiodicCorrelationTimeSweep(covMatrix, chronos)
%APERIODICCORRELATIONTIMESWEEP Sweeps the correlation time integral's upper
%limit to find the optimal estimation
%   @param covMatrix: POD's covariance matrix
%   %param chronos: the chronos of the resolved POD's states
%   @return: vector with the integrated correlation time up to a certain
%   point
%
% Author: Agustin PICARD, intern @ Scalian with Valentin RESSEGUIER as
% supervisor
%
N = size(covMatrix, 2);

% Estimate the aperiodic correlation function
corrFunction = estimateAperiodicCorrelation(smallScaleVelocityCov(covMatrix, chronos));

sweep = zeros(N, 1);
for i = 2 : N
    sweep(i) = 1 + 2 * sum(corrFunction(1 : i));
end

end

