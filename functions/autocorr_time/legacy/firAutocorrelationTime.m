function [tau] = firAutocorrelationTime(cov_v, bt)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

N = size(cov_v, 2);

% Estimate the small-scale velocity
cov_s = smallScaleVelocityCov(cov_v, bt);

% Generate the weight function
weights = firBuilder(N, floor(N / 2));

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

function weights = firBuilder(N, cutOffFreq)

weights = ifft([zeros(N - cutOffFreq, 1)', ones(2 * cutOffFreq, 1)', zeros(N - cutOffFreq, 1)']);
weights = abs(weights(1 : length(weights) / 2));
weights = weights / norm(weights);

end

