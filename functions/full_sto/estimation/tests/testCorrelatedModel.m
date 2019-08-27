%% Synthetic test for the noise terms' estimation formulas
clear all, close all, clc;

% Define some hyperparameters
n = 2; % number of modes
T = 1000; % number of simulation steps
% dt = 1e-7; % dt = 0.08;
dt = 0.08;

load('ILC_test.mat', 'ILC');
load('noises_test.mat', 'xi_xi_inf', 'theta_theta');

% Generate some ROM parameters randomly
tau = 100 * dt;
alpha_d = 100 * tau;
alpha_m = 1e-6;
I = ILC.tot.I;
% I = alpha_d .* randn(n, 1);
% I = zeros(n, 1);
L = ILC.tot.L;
% L = alpha_d .* randn(n, n);
% L = zeros(n, n);
C = ILC.tot.C;
% C = alpha_d .* randn(n, n, n);
% C = zeros(n, n, n);
theta_theta = reshape(theta_theta, [n * (n + 1), n * (n + 1)]);
% theta_theta = alpha_m .* rand((n + 1) * n, (n + 1) * n);
theta_theta = 0.5 .* (theta_theta + theta_theta');
theta_theta = theta_theta + n * (n + 1) * alpha_m * eye(n * (n + 1));
% [V, D] = eig(theta_theta);
% D = diag(D);
% D(D < 0) = 0;
% theta_theta = V * diag(D) * V';
chol_theta_theta = chol(theta_theta);
xi_xi = xi_xi_inf;
% xi_xi = alpha_m .* rand(n, n);
xi_xi = 0.5 .* (xi_xi + xi_xi');
xi_xi = xi_xi + n * alpha_m * eye(n);
% [V, D] = eig(xi_xi);
% D = diag(D);
% D(D < 0) = 0;
% xi_xi = V * diag(D) * V';
chol_xi_xi = chol(xi_xi);
theta_xi = zeros(n * (n + 1), n);

% Start the chronos at zero
bt = zeros(T, n);
bt_fv = zeros(T, n);
bt_m = zeros(T, n);
eta = zeros(T, n + 1, n);
Mi_ss = zeros(T, n);
spiral = zeros(T, 1);

% Simulate the evolution of the Chronos equations

for k = 1 : T - 1
    [bt(k + 1, :), bt_fv(k + 1, :), bt_m(k + 1, :), eta(k + 1, :, :) ...
        , Mi_ss(k + 1, :), spiral(k + 1, : ,:)] = ...
        evol_correlated_no_pchol(I, L, C, ...
                        chol_xi_xi, chol_theta_theta, tau, dt, bt(k, :), eta(k, :, :),...
                        spiral(k, :, :), Mi_ss(k, :), bt_fv(k, :), bt_m(k, :));
end

% Plot the resulting solutions
figure, plot(bt_fv), figure, plot(bt_m)

% Do the pertinent estimations
d2bt = diff(bt, 2, 1);
deta = diff(eta, 1, 1);
bt_x = cat(2, bt, ones(T, 1));
G_pq = bt_x' * bt_x ./ T;

% Beta
beta = zeros(n + 1, n, n + 1, n);
for p = 1 : n + 1
    for i = 1 : n
        for q = 1 : n + 1
            for j = 1 : n
                for k = 1 : T - 2
                    beta(p, i, q, j) = beta(p, i, q, j) + ...
                        bt_x(k, p) * d2bt(k, i) * deta(k, q, j) / G_pq(p, p);
                end
            end
        end
    end
end
beta = beta .* dt ./ (T - 2);

% theta_theta_est
theta_theta_est = zeros(n + 1, n, n + 1, n);
pinv_G = pinv(G_pq);
for i = 1 : n
    for q = 1 : n + 1
        for j = 1 : n
            theta_theta_est(:, i, q, j) = linsolve(G_pq, beta(:, i, q, j));
%             theta_theta_est(:, i, q, j) = pinv_G * beta(:, i, q, j);
        end
    end
end

% xi_xi_est
xi_xi_est = zeros(size(xi_xi));
for i = 1 : n
    for j = 1 : n
        kappa = 0;
        gamma = 0;
        for k = 1 : n
            kappa = kappa + d2bt(k, i) * d2bt(k, j);
        end
        kappa = kappa .* dt ./ (T - 2);
        for p = 1 : n + 1
            for q = 1 : n + 1
                gamma = gamma + G_pq(p, q) * theta_theta_est(p, i, q, j);
            end
        end
        xi_xi_est(i, j) = kappa - gamma;
    end
end

theta_theta_est = reshape(theta_theta_est, [n * (n + 1), n * (n + 1)]);
