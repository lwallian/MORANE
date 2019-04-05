function [tau] = LPFilter(tau_corr)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
N = length(tau_corr);

tau = 0;
weights = weightBuilder(N);

for j = 1 : N
    tau = tau + weights(j) * tau_corr(j);
end

tau = tau / N;

end


function weights = weightBuilder(N)

weights = zeros(N, 1);
weights(1) = 1;
numList(1) = 1;

for k = 2 : N
    num = 1 / (k - 1);
    numList = [numList, num];
    weights(k) = num / sum(numList);
end

end
