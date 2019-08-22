function cov_s = smallScaleVelocityCov(cov_v, bt)
% Calculates the small-scale velocity field given the complete field and
% the resolved chronos of the POD
%   @param cov_v: covariance matrix of the velocity field
%   @param bt: chronos of the POD
%   @return: the covariance matrix of the small-scale velocity field
%
% Author: Agustin PICARD, intern @ Scalian with Valentin RESSEGUIER as
% supervisor
%
global correlated_model;
N = size(cov_v, 2);

% Calculate the large and small scale covariance matrices
cov_w = zeros(size(cov_v));
if ~isempty(correlated_model) && correlated_model
    dbt = diff(bt, 1, 1);
    N = N - 1;
    for i = 1 : N
        for j = 1 : N
            cov_w(i,j) = dbt(i,:) * dbt(j,:)' / N;
        end
    end
else
    for i = 1 : N
        for j = 1 : N
            cov_w(i,j) = bt(i,:) * bt(j,:)' / N;
        end
    end
end

cov_s = (cov_v - cov_w);

end

