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

weights = weightBuilder(N);

tau = weights * tau_corr / N;

end


function weights = weightBuilder(N)

weights = [1.0, 1 ./ linspace(1, N - 1, N - 1)];
weights = weights ./ sum(weights);

end
