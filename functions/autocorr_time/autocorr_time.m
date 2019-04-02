function [tau] = autocorr_time(cov_v, bt)
%AUTOCORR_TIME Estimates de autocorrelation time given the covariance of
%the velocity field and of large scale velocity
%   @param is_big_data: boolean to vectorize or not
%   @param cov_v: covariance matrix of the velocity field
%   @param bt: chronos of the POD
N = size(cov_v, 2);

% Calculate the large and small scale covariance matrices
cov_w = zeros(N, N);
for i = 1 : N
    for j = 1 : N
        cov_w(i,j) = bt(i,:) * bt(j,:)' / N;
    end
end

cov_s = (cov_v - cov_w);

% Calculate the autocorrelation time as:
% -0.5 + sum(proy_sym(corr(tau))) / var_s
corr_s = 0;
var_s = trace(cov_s);

% c_prime = zeros(N - 1, 1); % for the test

for i = 2 : N
    c = 0;
    for j = 1 : N - i + 1
        c = c + cov_s(j,j+i-1);
    end
    %         c_prime(i - 1) = c; % for the test
    corr_s = corr_s + c .* N / (N - i + 1);
end

% figure, plot(c_prime ./ N), title('cov estimee') % for the test

tau = 1 + 2 * corr_s / var_s;

end
