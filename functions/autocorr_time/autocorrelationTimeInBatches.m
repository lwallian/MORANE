function [autocorrelationTime] = autocorrelationTimeInBatches(cov_v, bt, mode)
%AUTOCORRELATIONTIMEINBATCHES Estimates the autocorrelation time by
%partitioning the covariance matrix in chunks according to its periodicity.
%   @param cov_v: covariance matrix of the velocity field
%   @param bt: chronos basis of the resolved states
%   @param mode: string to choose whether the 'local' or 'global' variance is
%   used
%   @return: array with the autocorrelation time of each chunk
%
% Author: Agustin PICARD, intern @ Scalian with Valentin RESSEGUIER as
% supervisor
%
assert(strcmp(mode, 'global') || strcmp(mode, 'local'));

N = size(cov_v, 2);

% Estimate the small-scale velocity
cov_s = smallScaleVelocityCov(cov_v, bt);

% Estimate the period to know the size of the batches
period = periodicityFromAutocorrelation(cov_s);
% period = periodicityFromMatrix(cov_s);
% period = 5;
% period = N;

% Compute the autocorrelation time in batches
autocorrelationTime = zeros(ceil(N / period), 1);
periodIndex = 1;
currBatch{1} = 1;
currPosition = 1;
while currBatch{1} ~= 0
    currBatch = nextMatrixPeriod(cov_s, period, currPosition, mode);
    if currBatch{1} == 0
        break;
    end
    autocorrelationTime(periodIndex) = estimateAutocorrelationTime(currBatch);
    currPosition = currPosition + period;
    periodIndex = periodIndex + 1;
end

end

