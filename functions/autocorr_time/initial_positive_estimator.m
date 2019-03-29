function [tau] = initial_positive_estimator(cov_v, bt)
%INITIAL_POSITIVE_ESTIMATOR Implementation of the initial positive
%estimator for the autocorrelation time as in
%https://arxiv.org/abs/1011.0175 and Geyer's "Practical Markov Chain
%Monte Carlo".

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
c = 0;

for i = 2 : N
    c_old = c;
    c = 0;
    for j = 1 : N - i + 1
        c = c + cov_s(j,j+i-1);
    end
    if c_old + c < 0.0 % the chain is no longer reversible, irreducible and stationary
        break
    end
    corr_s = corr_s + c .* N / (N - i + 1);
end

tau = - 0.5 + corr_s / var_s;

end

