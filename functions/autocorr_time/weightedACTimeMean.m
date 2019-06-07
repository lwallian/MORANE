function [tau] = weightedACTimeMean(tau_corr)
%WEIGHTEDACTIMEMEAN Does a weighted mean for the correlation time estimate
% as per the following weighting function:
%   @param tau_corr: correlation time estimate as a function of the period.
%   @return: corresponding mean of the autocorrelation time.
%
% Author: Agustin PICARD, intern @ Scalian with Valentin RESSEGUIER as
% supervisor
%
N = length(tau_corr);

if N == 1
    tau = tau_corr;
    return
else
    weights = weightBuilder(N);
    tau = weights * tau_corr / N;
end

end


function weights = weightBuilder(N)

weights = [1.0, 1 ./ linspace(1, N - 1, N - 1)];
weights = weights ./ sum(weights);

end


function weights = butterBuilder(N, cutOffFreq)

weights = 1 ./ sqrt(1 + (linspace(0.0, N, N) ./ cutOffFreq).^(2 * N));

end

function weights = firBuilder(N, cutOffFreq)

weights = ifft([zeros(N - cutOffFreq, 1)', ones(2 * cutOffFreq, 1)', zeros(N - cutOffFreq, 1)']);
weights = abs(weights(1 : length(weights) / 2));

end
