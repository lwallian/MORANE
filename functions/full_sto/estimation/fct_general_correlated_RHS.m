function [R1, R2] = fct_general_correlated_RHS(param, bt, d2bt, sigma_ss)
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here

dt = param.dt;
N_tot = param.N_tot;
m = param.nb_modes;
nu = param.viscosity;
T = N_tot*dt;
M = param.M;
dX = param.dX;
MX = param.MX;
d = param.d;
lambda = param.lambda;

% The last two time steps are not used
N_tot = N_tot - 2;
T = T - 2 * dt;

% Estimate the terms using the orthogonality assumptions
[R1, R2] = fct_comp_correlated_RHS(param, bt, d2bt);

% As they are no longer valid, the statistics are going to be estimated
% through least squares
beta = bsxfun(@times, R1, 1 / lambda);

% Compute gamma
% First, we define G_pq
G_pq = mean(bt' * bt, 1); % check that it's the outer product given the dimensions, should be nxn

% We define psi_p


end

