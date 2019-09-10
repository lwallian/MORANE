%% Synthetic test for the noise terms' estimation formulas
clear all, close all, clc;

% Define some hyperparameters
n = 2; % number of modes
T = 10000; % number of simulation steps
T_cov = floor(T / 2);
% dt = 1e-7; % dt = 0.08;
dt = 0.008;
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
theta_theta = tau / n * 1000 .* eye(n * (n + 1));
theta_theta(n + 1, n + 1) = 0; theta_theta(end, end) = 0;
theta_theta = reshape(theta_theta, [n + 1, n, n + 1, n]);
theta_theta_c = theta_theta;
for p = 1 : n
    for i = 1 : n
        for q = 1 : n
            for j = 1 : n
                theta_theta(p, i, q, j) = 0.5 * (theta_theta_c(p, i, q, j) - theta_theta_c(i, p, q, j));
            end
        end
    end
end
theta_theta_c = theta_theta;
for p = 1 : n
    for i = 1 : n
        for q = 1 : n
            for j = 1 : n
                theta_theta(p, i, q, j) = 0.5 * (theta_theta_c(p, i, q, j) - theta_theta_c(p, i, j, q));
            end
        end
    end
end
theta_theta = reshape(theta_theta, [n * (n + 1), n * (n + 1)]);
% theta_theta = zeros((n + 1) * n, (n + 1) * n);
% theta_theta = reshape(theta_theta, [n * (n + 1), n * (n + 1)]);
% theta_theta = alpha_m .* rand((n + 1) * n, (n + 1) * n);
% theta_theta = 0.5 .* (theta_theta + theta_theta');
% theta_theta = theta_theta + n * (n + 1) * alpha_m * eye(n * (n + 1));
theta_theta = 0.5 * (theta_theta + theta_theta');
[V, D] = eig(theta_theta);
D = diag(D);
D(D < 0) = 0;
D = diag(D);
theta_theta = V * D * V';
chol_theta_theta = V * sqrt(D);
% chol_theta_theta = chol(theta_theta);
% xi_xi = zeros(n, n);
xi_xi = xi_xi_inf;
% xi_xi = alpha_m .* rand(n, n);
xi_xi = 0.5 .* (xi_xi + xi_xi');
% xi_xi = xi_xi + n * alpha_m * eye(n);
[V, D] = eig(xi_xi);
% xi_xi = V * sqrt(D) * V';
chol_xi_xi = V * sqrt(D);
theta_xi = zeros(n * (n + 1), n);

% Start the chronos at zero
bt = zeros(T, n, n_particles);
bt_fv = zeros(T, n, n_particles);
bt_m = zeros(T, n, n_particles);
bt_m(1, :, :) = randn(n, n_particles);
eta = zeros(T, n + 1, n, n_particles);
% eta(1, :, :, :) = randn(1, n + 1, n, n_particles);
Mi_ss = zeros(T, n, n_particles);
% Mi_ss(1, :, :) = randn(1, 1, n, n_particles);
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
figure, plot(reshape(mean_eta, T, [])), grid minor, title('eta')

t = 0 : dt : dt * (T_cov - 1);

% Verify that eta is effectively a frequency
eta_t = mean_eta(:, 1 : end - 1, :);
eta_t = permute(eta_t, [2 3 1]);
eig_eta = zeros(T, n);
for k = 1 : T
    eig_eta(k, :) = eig(eta_t(:, :, k));
end
any(isreal(eig_eta))

% Plot the modes in the complex plane
figure, hold on, plot(eig_eta(:, 1), 'or'), plot(eig_eta(:, 2), 'xb'), grid minor, hold off;

% Do the pertinent estimations
d2bt = diff(bt, 2, 1);
deta = diff(eta, 1, 1);
bt_x = cat(2, bt, ones(T, 1, n_particles));
G_pq = zeros(n + 1, n + 1, n_particles);
for k = 1 : T
    for p = 1 : n + 1
        for q = 1 : n + 1
            G_pq(p, q, :) = G_pq(p, q, :) + bt_x(k, p, :) .* bt_x(k, q, :);
        end
    end
end
G_pq = G_pq ./ T;
% G_pq = bt_x' * bt_x ./ T;

% Beta
beta = zeros(n + 1, n, n + 1, n, n_particles);
for l = 1 : n_particles
    for p = 1 : n + 1
        for i = 1 : n
            for q = 1 : n + 1
                for j = 1 : n
                    for k = 1 : T - 2
                        beta(p, i, q, j, l) = beta(p, i, q, j, l) + ...
                            bt_x(k, p, l) * d2bt(k, i, l) * deta(k, q, j, l) / G_pq(p, p, l);
                    end
                end
            end
        end
    end
end
beta = beta .* dt ./ (T - 2);

% theta_theta_est
theta_theta_est = zeros(n + 1, n, n + 1, n, n_particles);
for k = 1 : n_particles
    pinv_G = pinv(G_pq(:, :, k));
    for i = 1 : n
        for q = 1 : n + 1
            for j = 1 : n
                theta_theta_est(:, i, q, j, k) = linsolve(G_pq(:, :, k), beta(:, i, q, j, k));
%                 theta_theta_est(:, i, q, j, k) = pinv_G * beta(:, i, q, j);
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
