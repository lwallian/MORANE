%% Synthetic test for the martingale integration of the correlated model
clear all, close all, clc;

% Define some hyperparameters
n = 2; % number of modes
T = 10000; % number of simulation steps
T_cov = floor(T / 2);
% dt = 1e-7; % dt = 0.08;
dt = 0.008;
n_particles = 100;

load('noises_test.mat', 'xi_xi_inf', 'theta_theta');

% Generate some ROM parameters randomly
tau = 20 * dt;
alpha_d = 100 * tau;
alpha_m = 1e-6;
% theta_theta = zeros((n + 1) * n, (n + 1) * n);
% chol_theta_theta = zeros((n + 1) * n, (n + 1) * n);
% theta_theta = alpha_m .* rand((n + 1) * n, (n + 1) * n);
theta_theta = reshape(theta_theta, [n * (n + 1), n * (n + 1)]);
theta_theta = 0.5 .* (theta_theta + theta_theta');
theta_theta = theta_theta + n * (n + 1) * alpha_m * eye(n * (n + 1));
chol_theta_theta = chol(theta_theta);
% xi_xi = zeros(n, n);
% chol_xi_xi = zeros(n, n);
% xi_xi = alpha_m .* rand(n, n);
xi_xi = xi_xi_inf;
xi_xi = 0.5 .* (xi_xi + xi_xi');
xi_xi = xi_xi + n * alpha_m * eye(n);
chol_xi_xi = chol(xi_xi);
theta_xi = zeros(n * (n + 1), n);

% Start the chronos at zero
bt_m = zeros(T, n, n_particles);
eta = zeros(T, n + 1, n, n_particles);
eta(1, :, :, :) = randn(1, n + 1, n, n_particles);
Mi_ss = zeros(T, n, n_particles);
Mi_ss(1, :, :) = randn(1, 1, n, n_particles);
spiral = zeros(T, 1, n_particles);
spiral(1, :, :) = randn(1, 1, n_particles);

% Simulate the evolution of the Chronos equations

for k = 1 : T - 1
    [bt_m(k + 1, :, :), eta(k + 1, :, :, :) ...
        , Mi_ss(k + 1, :, :), spiral(k + 1, : ,:, :)] = ...
        evol_correlated_martingale_no_pchol(chol_xi_xi, chol_theta_theta, tau, dt, eta(k, :, :, :),...
                        spiral(k, :, :, :), Mi_ss(k, :, :), bt_m(k, :, :));
end

% Plot the resulting solutions
figure, plot(mean(bt_m, 3)), title('Martingale'), grid minor;

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
for k = 1 : (T_cov - 1)
    for j = (T - T_cov) : (T - 1 - k + 1)
        spiral_cov(k) = spiral_cov(k) + ...
            mean_spiral(j) * mean_spiral(k + j);
    end
    spiral_cov(k) = spiral_cov(k) / (T - T_cov - k);
end
theo_spiral_cov = dt .* exp(- t / tau);
figure, hold on;
plot(spiral_cov), title('Spiral correlation'), grid minor;
plot(theo_spiral_cov);

% Verify Mi_ss' moments 1 and 2
Mi_ss_mean = mean(mean_Mi_ss, 1)
Mi_ss_cov = zeros(T_cov, n);
for i = 1 : n
    for k = 1 : (T_cov - 1)
        for j = (T - T_cov) : (T - 1 - k + 1)
            Mi_ss_cov(k, i) = Mi_ss_cov(k, i) + ...
                mean_Mi_ss(j, i) * mean_Mi_ss(k + j, i);
        end
        Mi_ss_cov(k, i) = Mi_ss_cov(k, i) / (T - T_cov - k);
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
        for k = 1 : (T_cov - 1)
            for l = (T - T_cov) : (T - 1 - k + 1)
                eta_cov(k, i, j) = eta_cov(k, i, j) + ...
                    mean_eta(l, i, j) * mean_eta(l + k, i, j);
%                 (mean_eta(l, i, j) - eta_mean(:, i, j)) * (mean_eta(l + k, i, j) - eta_mean(:, i, j));
            end
            eta_cov(k, i, j) = eta_cov(k, i, j) / (T - T_cov - k);
        end
    end
end
theo_eta_cov = theta_theta(2, 2) * tau / 2 * dt .* exp(- t / tau);
figure, hold on;
plot(eta_cov(:, 2, 2)), title('Eta correlation'), grid minor;
plot(theo_eta_cov);
