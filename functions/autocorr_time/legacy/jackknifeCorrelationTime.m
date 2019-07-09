function [tau] = jackknifeCorrelationTime(cov_v, bt, mode)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
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
NChunks = ceil(N / period);
jackAutocorr = zeros(NChunks, 1);
for i = 1 : NChunks
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
        if periodIndex == i
            currPosition = currPosition + period;
        end
    end
    jackAutocorr(i) = mean(autocorrelationTime);
end

tau = mean(jackAutocorr);

end

