function [tau] = autocorr_time_sweep(cov_v, bt)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
N = size(cov_v, 2);
lag_f = N;

% Calculate the large and small scale covariance matrices
cov_w = zeros(N, N);
for i = 1 : N
    for j = 1 : N
        cov_w(i,j) = bt(i,:) * bt(j,:)' / N;
    end
end

cov_s = (cov_v - cov_w);
var_s = trace(cov_s);

tau = zeros(N - lag_f, 1);

for k = 2 : lag_f
    tau(k) = autocorr_time_slice(cov_s, var_s, k);
end

figure, plot(tau);
title('Autocorrelation time estimation')
xlabel('Time lag'), ylabel('Autocorrelation time estimate');
grid minor;

end


function tau = autocorr_time_slice(cov_s, var_s, lag)
% Calculate the autocorrelation time as:
% 1 + 2 * sum(proy_sym(corr(tau))) / var_s
corr_s = 0;

for i = 2 : lag
    c = 0;
    for j = 1 : lag - i + 1
        c = c + cov_s(j,j+i-1);
    end
    corr_s = corr_s + c .* lag / (lag - i + 1);
end

tau = 1 + 2 * corr_s / var_s;

end

