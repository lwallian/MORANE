function [filteredSignal, optimalTaps] = minimalVarianceLMS(refSignal, maxTaps, delay)
%MINIMALVARIANCELMS Performs an LMS filtering periodic signal cancelation
%given a delay and sweeping the amount of filter taps to find the lowest
%variation at the beginning of the signal.
%   @param refSignal: the reference signal we want to filter
%   @param maxTaps: the maximum amount of taps to sweep
%   @param delay: constant delay used in the LMS cancelation
%   @return filteredSignal: the filtered signal
%   @return optimalTaps: the optimal number of taps according to that
%   metric
%
% Author: Agustin PICARD, intern @ Scalian with Valentin RESSEGUIER as
% supervisor
%

% Apply the filter and check the variance
filteredVariation = zeros(maxTaps, 1);
for i = 1 : maxTaps
    filteredCorrelation = LMSFilter(refSignal, i, delay);
    if length(refSignal) < 5 * delay
        filteredVariation(i) = norm(diff(filteredCorrelation));
    else
        filteredVariation(i) = norm(diff(filteredCorrelation(1 : 5 * delay)));
    end
end

% Choose the one with the minimum variance and use that filter
[~, optimalTaps] = min(filteredVariation);
filteredSignal = LMSFilter(refSignal, optimalTaps, delay);

end

