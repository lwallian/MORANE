%% Synthetic test for the noise terms' estimation formulas
clear all, close all, clc;

% Define some hyperparameters
n = 2; % number of modes
T = 10000; % number of simulation steps
T_cov = floor(T / 2);
% dt = 1e-7; % dt = 0.08;
dt = 0.08;
n_particles = 100;

load('ILC_test.mat', 'ILC');
load('noises_test.mat', 'xi_xi_inf', 'theta_theta');

% Generate some ROM parameters randomly
tau = 20 * dt;
alpha_d = 100 * tau;
alpha_m = 1e-6;
% I = ILC.tot.I;
% I = alpha_d .* randn(n, 1);
I = zeros(n, 1);
% L = ILC.tot.L;
% L = alpha_d .* randn(n, n);
L = zeros(n, n);
% C = ILC.tot.C;
% C = alpha_d .* randn(n, n, n);
C = zeros(n, n, n);
% theta_theta = zeros((n + 1) * n, (n + 1) * n);
theta_theta = reshape(theta_theta, [n * (n + 1), n * (n + 1)]);
% theta_theta = alpha_m .* rand((n + 1) * n, (n + 1) * n);
theta_theta = 0.5 .* (theta_theta + theta_theta');
theta_theta = theta_theta + n * (n + 1) * alpha_m * eye(n * (n + 1));
% [V, D] = eig(theta_theta);
% D = diag(D);
% D = 10 .* D;
% D(D < 0) = 0;
% theta_theta = V * diag(D) * V';
chol_theta_theta = chol(theta_theta);
% xi_xi = zeros(n, n);
xi_xi = xi_xi_inf;
% xi_xi = alpha_m .* rand(n, n);
xi_xi = 0.5 .* (xi_xi + xi_xi');
xi_xi = xi_xi + n * alpha_m * eye(n);
% [V, D] = eig(xi_xi);
% D = diag(D);
% D = 10 .* D;
% D(D < 0) = 0;
% xi_xi = V * diag(D) * V';
chol_xi_xi = chol(xi_xi);
theta_xi = zeros(n * (n + 1), n);

% Start the chronos at zero
bt = zeros(T, n, n_particles);
bt_fv = zeros(T, n, n_particles);
bt_m = zeros(T, n, n_particles);
eta = zeros(T, n + 1, n, n_particles);
eta(1, :, :, :) = randn(1, n + 1, n, n_particles);
Mi_ss = zeros(T, n, n_particles);
Mi_ss(1, :, :) = randn(1, 1, n, n_particles);
spiral = zeros(T, 1, n_particles);
spiral(1, :, :) = randn(1, 1, n_particles);

% Simulate the evolution of the Chronos equations

for k = 1 : T - 1
    [bt(k + 1, :, :), bt_fv(k + 1, :, :), bt_m(k + 1, :, :), eta(k + 1, :, :, :) ...
        , Mi_ss(k + 1, :, :), spiral(k + 1, : ,:, :)] = ...
        evol_correlated_no_pchol(I, L, C, ...
                        chol_xi_xi, chol_theta_theta, tau, dt, bt(k, :, :), eta(k, :, :, :),...
                        spiral(k, :, :, :), Mi_ss(k, :, :), bt_fv(k, :, :), bt_m(k, :, :));
end

% Plot the resulting solutions
figure, plot(mean(bt_fv, 3)), title('Finite variation'), grid minor;
figure, plot(mean(bt_m, 3)), title('Martingale'), grid minor;
figure, plot(mean(bt, 3)), title('Total'), grid minor;

% Plot the different stochastic mean variables
mean_spiral = mean(spiral, 3);
mean_Mi_ss = mean(Mi_ss, 3);
mean_eta = mean(eta, 4);
figure, plot(mean_spiral), title('spiral'), grid minor;
figure, plot(mean_Mi_ss), title('Mi_{ss}'), grid minor;

t = 0 : dt : dt * (T_cov - 1);

% Verify the spiral's moments 1 and 2
spiral_mean = mean(mean(spiral, 3), 1)
spiral_cov = zeros(T_cov, 1);
for k = 1 : T_cov - 1
    for j = T_cov : T - 1 - k + 1
        spiral_cov(k) = spiral_cov(k) + ...
            mean_spiral(j) * mean_spiral(k + j);
    end
    spiral_cov(k) = spiral_cov(k) / (T_cov - k);
end
theo_spiral_cov = dt .* exp(- t / tau);
figure, hold on;
plot(spiral_cov), title('Spiral correlation'), grid minor;
plot(theo_spiral_cov);

% Verify Mi_ss' moments 1 and 2
Mi_ss_mean = mean(mean_Mi_ss, 1)
Mi_ss_cov = zeros(T_cov, n);
for i = 1 : n
    for k = 1 : T_cov - 1
        for j = T_cov : T - 1 - k + 1
            Mi_ss_cov(k, i) = Mi_ss_cov(k, i) + ...
                mean_Mi_ss(j, i) * mean_Mi_ss(k + j, i);
        end
        Mi_ss_cov(k, i) = Mi_ss_cov(k, i) / (T_cov - k);
    end
end
theo_Mi_ss_cov = tau ./ 4 .* xi_xi(1, 1) .* dt .* exp(- 2 * t / tau);
figure, hold on;
plot(Mi_ss_cov(:, 1)), title('Mi_{ss} correlation'), grid minor;
plot(theo_Mi_ss_cov);


% Verify eta's moments 1 and 2
eta_mean = mean(mean_eta, 1)
eta_cov = zeros(T_cov, n + 1, n);
for i = 1 : n + 1
    for j = 1 : n
        for k = 1 : T_cov - 1
            for l = T_cov : T - 1 - k + 1
                eta_cov(k, i, j) = eta_cov(k, i, j) + ...
                    (mean_eta(l, i, j) - eta_mean(:, i, j)) * (mean_eta(l + k, i, j) - eta_mean(:, i, j));
            end
            eta_cov(k, i, j) = eta_cov(k, i, j) / (T_cov - k);
        end
    end
end
theo_eta_cov = theta_theta(1, 1) * tau / 2 * dt .* exp(- t / tau);
figure, hold on;
plot(eta_cov(:, 1, 1)), title('Eta correlation'), grid minor;
plot(theo_eta_cov);

% Theoretical variance values
theo_var_eta = theta_theta * tau / 2 * dt;
theo_var_Mi_ss = tau / 4 * (chol_xi_xi * chol_xi_xi') * dt * dt;
theo_var_spiral = 1 * dt;

% Empirical variance values
emp_var_eta = var(mean_eta, 0, 1);
emp_var_Mi_ss = var(mean_Mi_ss, 0, 1);
emp_var_spiral = var(mean_spiral, 0, 1);

% Do the pertinent estimations
d2bt = diff(bt, 2, 1);
deta = diff(eta, 1, 1);
bt_x = cat(2, bt, ones(T, 1, n_particles));
G_pq = zeros(n + 1, n + 1, n_particles);
for k = 1 : T
    for i = 1 : n + 1
        for j = 1 : n + 1
            G_pq(i, j, :) = G_pq(i, j, :) + bt_x(k, i, :) .* bt_x(k, j, :);
        end
    end
end
% G_pq = bsxfun(@times, bt_x, bt_x);
% G_pq = sum(G_pq, 1);
G_pq = G_pq ./ T;
% G_pq = bt_x' * bt_x ./ T;

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
theta_theta_est = zeros(n + 1, n, n + 1, n, n_particles);
% pinv_G = pinv(G_pq);
for k = 1 : n_particles
    for i = 1 : n
        for q = 1 : n + 1
            for j = 1 : n
                theta_theta_est(:, i, q, j, k) = linsolve(G_pq(:, :, k), beta(:, i, q, j));
                %             theta_theta_est(:, i, q, j) = pinv_G * beta(:, i, q, j);
            end
        end
    end
end

% xi_xi_est
xi_xi_est = zeros([size(xi_xi), n_particles]);
for k = 1 : n_particles
    for i = 1 : n
        for j = 1 : n
            kappa = d2bt(:, i, k)' * d2bt(:, j, k) * dt / (T - 2);
            gamma = 0;
            for p = 1 : n + 1
                for q = 1 : n + 1
                    gamma = gamma + G_pq(p, q, k) * theta_theta_est(p, i, q, j, k);
                end
            end
            xi_xi_est(i, j, k) = kappa - gamma;
        end
    end
end

xi_xi_mean = mean(xi_xi_est, 3);
theta_theta_mean = mean(reshape(theta_theta_est, [n * (n + 1), n * (n + 1), n_particles]), 3);
