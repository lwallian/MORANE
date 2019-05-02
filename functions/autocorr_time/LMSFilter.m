function [outSignal] = LMSFilter(refSignal, taps, delay)
%LMSFILTER Least Squares filter to remove the periodic part of
%the signal in the autocorrelation function
%   @param refSignal: reference signal namely the autocorrrelation function
%   @param taps: filter order
%   @param delay: delay of the input signal wrt the reference for
%   periodicity elimination
%   @return: filtered autocorrelation function
%
% Author: Agustin PICARD, intern @ Scalian with Valentin RESSEGUIER as
% supervisor
%
N = length(refSignal);

% Initialize filter
h = zeros(taps, 1);
inputSignal = [zeros(delay, 1)', circshift(refSignal, delay)'];
inputSignal = inputSignal(1 : N);

outSignal = zeros(N, 1);
for i = 1 : N - taps - 1
    inputSlice = inputSignal(i : i + taps - 1)';
    outSignal(i) = refSignal(i) - h' * inputSlice;
%     h = h + outSignal(i) * inputSlice / norm(inputSlice);
    h = h + 0.01 * outSignal(i) * inputSlice;
end

end

