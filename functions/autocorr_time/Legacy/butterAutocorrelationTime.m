function [tau] = butterAutocorrelationTime(cov_v, bt)
%BUTTERAUTOCORRELATIONTIME Estimates the autocorrelation time with the
%usual formula except a weight for each coefficient given by a Butterworth
%filter
%   @param cov_v: Covariance matrix of the velocity field
%   @param bt: Chronos of the resolved states
%   @return: the autocorrelation time estimation
%
% Author: Agustin PICARD, intern @ Scalian with Valentin RESSEGUIER as
% supervisor
%
N = size(cov_v, 2);

% Estimate the small-scale velocity
cov_s = smallScaleVelocityCov(cov_v, bt);

% Generate the weight function
weights = butterBuilder(N, N / 2);

% Estimate the correlation time
autocorrelation = zeros(N - 1, 1);
var_s = trace(cov_s);
for i = 2 : N
    c = 0;
    for j = 1 : N - i + 1
        c = c + cov_s(j,j+i-1);
    end
    autocorrelation(i - 1) = c * weights(i - 1) * N / (N - i + 1);
end

autocorrelation = autocorrelation ./ var_s;

tau = 1 + 2 * sum(autocorrelation);

end

function weights = butterBuilder(N, cutOffFreq)

weights = 1 ./ sqrt(1 + (linspace(0.0, N, N) ./ cutOffFreq).^(2 * N));
weights = weights / norm(weights);

end

