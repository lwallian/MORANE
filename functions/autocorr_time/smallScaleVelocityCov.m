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
N = size(cov_v, 2);

% Calculate the large and small scale covariance matrices
cov_w = (1/ N) * (bt * bt') ;

cov_s = (cov_v - cov_w);

end

