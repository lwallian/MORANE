function [outputSignal] = RLSFilter(refSignal, taps, delay, forgetFactor)
%RLSFilter Recursive Least Squares filter to remove the periodic part of
%the signal in the autocorrelation function
%   @param refSignal: reference signal namely the autocorrrelation function
%   @param taps: filter order
%   @param delay: delay of the input signal wrt the reference for
%   periodicity elimination
%   @param forgetFactor: forget factor of the algorithm
%   @return: filtered autocorrelation function
%
% Author: Agustin PICARD, intern @ Scalian with Valentin RESSEGUIER as
% supervisor
%
N = length(refSignal);
invForget = 1 / forgetFactor;

% Initialize filter
h = zeros(taps, 1);
P = eye(taps);
inputSignal = [zeros(delay, 1)', circshift(refSignal, delay)'];
inputSignal = inputSignal(1 : N);

outputSignal = zeros(N, 1);
for i = 1 : N - taps
    inputSlice = inputSignal(i : i + taps - 1)';
    outputSignal(i) = refSignal(i) - h' * inputSlice;
    gain = P * inputSlice * pinv(forgetFactor + inputSlice' * P * inputSlice);
    P = invForget * P - gain * inputSlice' * invForget * P;
    h = h + outputSignal(i) * gain;
end

end

