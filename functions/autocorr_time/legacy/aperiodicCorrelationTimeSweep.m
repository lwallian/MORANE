function [sweep] = aperiodicCorrelationTimeSweep(covMatrix, chronos)
%APERIODICCORRELATIONTIMESWEEP Sweeps the correlation time integral's upper
%limit to find the optimal estimation
%   @param covMatrix: POD's covariance matrix
%   @param chronos: the chronos of the resolved POD's states
%   @return: vector with the integrated correlation time up to a certain
%   point
%
% Author: Agustin PICARD, intern @ Scalian with Valentin RESSEGUIER as
% supervisor
%
N = size(covMatrix, 2);
% N = 10000;
cov_s = smallScaleVelocityCov(covMatrix, chronos);
var_s = trace(cov_s);
periodicPart = estimatePeriodicCorrelation(cov_s);

sweep = zeros(N - 1, 1);
aperiodic = nextAperiodicACSlice(cov_s, 1, periodicPart);
sweep(1) =  1 + 2 * aperiodic ./ var_s;
for i = 2 : N - 1
%     aperiodic = [aperiodic ./ (i - 1), nextAperiodicACSlice(cov_s, i, periodicPart)] .* i;
    aperiodic = [aperiodic ./ N, nextAperiodicACSlice(cov_s, i, periodicPart)] .* N;
    sweep(i - 1) = 1 + 2 * sum(aperiodic) / var_s;
end

end


function slice = nextAperiodicACSlice(cov_s, position, periodicPart)

periodicIndex = mod(position, length(periodicPart)) + 1;
slice = mean(diag(cov_s, position)) - periodicPart(periodicIndex);

end

