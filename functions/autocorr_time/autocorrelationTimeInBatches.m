function [autocorrelationTime] = autocorrelationTimeInBatches(cov_v, bt)
%AUTOCORRELATIONTIMEINBATCHES Estimates the autocorrelation time by
%partitioning the covariance matrix in chunks according to its periodicity.
%   @param cov_v: covariance matrix of the velocity field
%   @param bt: chronos basis of the resolved states
%
% Author: Agustin PICARD, intern @ Scalian with Valentin RESSEGUIER as
% supervisor
%
N = size(cov_v, 2);

% Estimate the small-scale velocity
cov_s = smallScaleVelocityCov(cov_v, bt);

% Estimate the period to know the size of the batches
period = estimateCovariancePeriod(cov_s);

% Compute the autocorrelation time in batches
autocorrelationTime = zeros(ceil(N / period), 1);
periodIndex = 1;
currBatch{1} = 1;
currPosition = 1;
while currBatch{1} ~= 0
    currBatch = nextMatrixPeriod(cov_s, period, currPosition);
    autocorrelationTime(periodIndex) = estimateAutocorrelationTime(currBatch);
    currPosition = currPosition + period;
    periodIndex = periodIndex + 1;
end

end

function cov_s = smallScaleVelocityCov(cov_v, bt)

N = size(cov_v, 2);

% Calculate the large and small scale covariance matrices
cov_w = zeros(size(cov_v));
for i = 1 : N
    for j = 1 : N
        cov_w(i,j) = bt(i,:) * bt(j,:)' / N;
    end
end

cov_s = (cov_v - cov_w);

end

