function [tau] = autocorr_time_sweep(cov_v, bt)
%AUTOCORR_TIME_SWEEP Sweeps the vector length of autocorrelation to observe
%how the variance and mean of the estimator changes along with it.
%   @param cov_v: covariance matrix of the velocity field
%   @param bt: chronos basis of the resolved states
%
% Author: Agustin PICARD, intern @ Scalian with Valentin RESSEGUIER as
% supervisor
%
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

