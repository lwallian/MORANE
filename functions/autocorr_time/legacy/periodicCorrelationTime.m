function [autocorrelationTime] = periodicCorrelationTime(cov_v, bt, dt)
%PERIODICCORRELATIONTIME Estimates the correlation time using the Strouhal
%period to cut the covariance matrix in pieces
%   @param cov_v: covariance matrix
%   @param bt: chronos of the POD
%   @param dt: model's sampling period
%   @return: estimation of the correlation time
%
% Author: Agustin PICARD, intern @ Scalian with Valentin RESSEGUIER as
% supervisor
%
N = size(cov_v, 2);

% Estimate the small-scale velocity
cov_s = smallScaleVelocityCov(cov_v, bt);

% Set the period as the Strouhal period (0.2 Hz)
period = 5 / dt;

% Compute the autocorrelation time in batches
autocorrelationTime = zeros(ceil(N / period), 1);
periodIndex = 1;
currBatch{1} = 1;
currPosition = 1;
while currBatch{1} ~= 0
    currBatch = nextMatrixPeriod(cov_s, period, currPosition, 'global');
    if currBatch{1} == 0
        break;
    end
    autocorrelationTime(periodIndex) = estimateAutocorrelationTime(currBatch);
    currPosition = currPosition + period;
    periodIndex = periodIndex + 1;
end

end

