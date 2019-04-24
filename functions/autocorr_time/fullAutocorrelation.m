function [autocorrelation] = fullAutocorrelation(cov_s)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
N = size(cov_s, 2);
autocorrelation = zeros(N - 1, 1);
var_s = trace(cov_s);

autocorrelation(1) = var_s;
for i = 2 : N
    c = 0;
    for j = 1 : N - i + 1
        c = c + cov_s(j,j+i-1);
    end
    autocorrelation(i) = c * N / (N - i + 1);
end

autocorrelation = autocorrelation ./ var_s;

end

