%% Test the function for estimating the correlated model's noises
clear all, close all, clc;

% Define simulation parameters
n = 2; % number of modes
Temps_integ = 80
dt = 0.0008
T = floor(Temps_integ / dt) % number of simulation steps
n_particles = 1;
time = 0 : dt : dt * (T - 1);

% Load the determinisitic ROM parameters
load('ILC_test.mat', 'ILC','bt_tot');
I = ILC.tot.I;
L = ILC.tot.L;
C = ILC.tot.C;

% Generate some ROM parameters randomly
coef_expl = 1e-2
tau_min = 24 / 10
tau = tau_min

chol_theta_theta = sqrt( coef_expl * (n/(tau*tau_min^2)) ) * randn(n * (n + 1));
theta_theta =  chol_theta_theta * chol_theta_theta' ;
theta_theta = 0.5 * (theta_theta + theta_theta');
[V, D] = eig(theta_theta);
D = diag(D);
D(D < 0) = 0;
D = diag(D);
theta_theta = V * D * V';
chol_theta_theta = V * sqrt(D);

bt = zeros(T, n, n_particles);
bt(1, :, :) = repmat(bt_tot(1, :), [1, 1, n_particles]);

chol_xi_xi = sqrt((2 * mean(bt(1, :, :) / tau)).^2 / tau_min) * randn(n);
xi_xi =  chol_xi_xi * chol_xi_xi' ;
xi_xi = 0.5 .* (xi_xi + xi_xi');
[V, D] = eig(xi_xi);
D = diag(D);
D(D < 0) = 0;
D = diag(D);
xi_xi = V * D * V';
chol_xi_xi = V * sqrt(D);
theta_xi = zeros(n * (n + 1), n);

bt_fv = zeros(T, n, n_particles);
bt_m = zeros(T, n, n_particles);
eta = zeros(T, n + 1, n, n_particles);
Mi_ss = zeros(T, n, n_particles);
spiral = zeros(T, 1, n_particles);

% Integrate the chronos
for k = 1 : T - 1
    [bt(k + 1, :, :), bt_fv(k + 1, :, :), bt_m(k + 1, :, :), eta(k + 1, :, :, :) ...
        , Mi_ss(k + 1, :, :), spiral(k + 1, : ,:, :)] = ...
        evol_correlated_no_pchol(I, L, C, ...
        chol_xi_xi, chol_theta_theta, tau, dt, bt(k, :, :), eta(k, :, :, :),...
        spiral(k, :, :, :), Mi_ss(k, :, :), bt_fv(k, :, :), bt_m(k, :, :));
end

% Estimate the parameters with the corresponding function
[theta_theta_est, xi_xi_est] = estimate_noise_matrices_orthogonal(bt, eta, n, T, dt, n_particles, false);

% Estimate the error made while estimating
err_theta = (theta_theta - theta_theta_est).^2 / mean((theta_theta(:)).^2);
sq_mean_err_theta = sqrt(mean(err_theta(:)))
on_sq_N_ech = 1/sqrt( (T*dt)/tau )

err_xi_xi = (xi_xi - xi_xi_est).^2 / mean((xi_xi(:)).^2);
sq_mean_err_xi_xi = sqrt(mean(err_xi_xi(:)))
on_sq_N_ech = 1/sqrt( (T*dt)/tau )
