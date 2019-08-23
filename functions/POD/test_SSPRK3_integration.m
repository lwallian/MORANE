%% Test SSPRK3 with Ornstein-Uhlenbeck
%% Test of deterministic component
clear all, close all, clc;

% Simulation parameters
ti = 0;
tf = 1000;
dt = 0.01;
n_realizations = 1;
T = (tf - ti) / dt;

% Evolution equations
drift = @(A, Xt) -A * Xt;
martingale = @(C) C * sqrt(dt) * randn(n_realizations, 1);
stochastic_evolve = @(A, Xt, C) drift(A, Xt) + martingale(C);

% The Ornstein-Uhlenbeck process has the following form:
% dX = -A * X dt + C dBt
% Solution should be: X = exp(-A * t) + int_0^t (exp(-A * s) * C dBs)
A = 1.0 .* eye(n_realizations);
C = 1.0 .* eye(n_realizations);
X0 = 10.0 .* ones(n_realizations, 1);
Xt_RK = zeros(n_realizations, T);
Xt_RK(:, 1) = X0;
k = 2;

for t = ti : dt: tf
    % SSPRK3 integration scheme with small dt for stability
%     k1 = stochastic_evolve(A, Xt(:, k - 1), C);
%     u1 = Xt(:, k - 1) + dt * k1;
%     k2 = stochastic_evolve(A, u1, C);
%     u2 = 3 / 4 * Xt(:, k - 1) + u1 / 4 + dt * k2 / 4;
%     k3 = stochastic_evolve(A, u2, C);
%     Xt(:, k) = Xt(:, k - 1) / 3 + (2 / 3) * (u2 + dt * k3);
%     k = k + 1;
    sigma = martingale(C);
    k1 = drift(A, Xt_RK(:, k - 1));
%     u1 = Xt(:, k - 1) + dt * k1 + martingale(C);
    u1 = Xt_RK(:, k - 1) + dt * k1 + sigma;
    k2 = drift(A, u1);
    u2 = 3 / 4 * Xt_RK(:, k - 1) + (u1 / 4) + (dt * k2 / 4 + sigma / 4);
%     u2 = 3 / 4 * Xt(:, k - 1) + (u1 / 4) + (dt * k2 / 4 + martingale(C) / 4);
    k3 = drift(A, u2);
%     Xt(:, k) = (Xt(:, k - 1) / 3) + (2 / 3) * (u2 + dt * k3 + martingale(C));
    Xt_RK(:, k) = (Xt_RK(:, k - 1) / 3) + (2 / 3) * (u2 + dt * k3 + sigma);
    k = k + 1;
end

% Through the plot we verify the decay time (normally with a small noise
% coefficient)
t = ti : dt : tf;
figure, plot(t, Xt_RK(:, 1 : end - 1));
grid minor, title('Ornstein Uhlenbeck realizations');

Xt_mean = mean(Xt_RK, 1);
figure, plot(t, Xt_mean(1 : end - 1));
grid minor, title('Mean realization')

dXt = diff(Xt_RK);
noise = sqrt(dt) * randn(length(Xt_RK), 1);
figure, hold on;
plot(noise), plot(dXt), grid minor;
hold off;

%% Test of stochastic component
% For the stochastic part, we test by checking the correlation between
% samples
corr_func = zeros(n_realizations, n_realizations, T / 2);
mu = mean(Xt_mean(T / 2 : end));

% Empirical correlation function
for k = 1 : T / 2
    i = 1;
    for j = T / 2 : T - k + 1
        R_x(:, :, i) = (Xt_RK(:, j) - mu) * (Xt_RK(:, j + k - 1) - mu)';
        i = i + 1;
    end
    corr_func(:, :, k) = mean(R_x, 3);
    clear R_x;
end

% Theorical correlation function for an OU process
corr_theo = zeros(n_realizations, n_realizations, T / 2);
i = 1;
for t = 0 : dt : tf / 2 - 1
    corr_theo(:, :, i) = C * C' * exp(-t .* A) * pinv(A + A');
    i = i + 1;
end

figure, hold on;
t = 0 : dt : tf / 2; t = t(1 : end - 1);
plot(t, squeeze(corr_func)), plot(t, squeeze(corr_theo)), grid minor;
% plot(t, corr_func), plot(t, corr_theo), grid minor;
title('Theoretical and empirical correlation functions');
hold off;

%% Test Euler-Maruyama vs SSPRK3
clear all, close all, clc;

% Simulation parameters
ti = 0;
tf = 1000;
dt = 0.1;
n_realizations = 100;
T = (tf - ti) / dt;

% We'll test with an equation of the form:
% dX = X dBt
% Solution should be: X = exp(-A * t) + int_0^t (exp(-A * s) * C dBs)
drift = @(A, Xt) -A * Xt;
martingale = @(C) C * sqrt(dt) * randn(n_realizations, 1);

% The Ornstein-Uhlenbeck process has the following form:
% dX = -A * X dt + C dBt
% Solution should be: X = exp(-A * t) + int_0^t (exp(-A * s) * C dBs)
A = 0.1 .* eye(n_realizations);
C = 1.0 .* eye(n_realizations);
X0 = 10.0 .* ones(n_realizations, 1);
Xt_RK = zeros(n_realizations, T);
Xt_RK(:, 1) = X0;
Xt_EM = zeros(n_realizations, T);
Xt_EM(:, 1) = X0;
k = 2;

for t = ti : dt: tf
    % SSPRK3 integration scheme with small dt for stability
    sigma = martingale(C);
    k1 = drift(A, Xt_RK(:, k - 1));
    u1 = Xt_RK(:, k - 1) + dt * k1 + sigma;
    k2 = drift(A, u1);
    u2 = 3 / 4 * Xt_RK(:, k - 1) + (u1 / 4) + (dt * k2 / 4 + sigma / 4);
    k3 = drift(A, u2);
    Xt_RK(:, k) = (Xt_RK(:, k - 1) / 3) + (2 / 3) * (u2 + dt * k3 + sigma);
    % Euler Maruyama
    Xt_EM(:, k) = Xt_EM(:, k - 1) + dt * drift(A, Xt_EM(:, k - 1)) + sigma;
    k = k + 1;
end

% Plot the two means
Xt_RK_mean = mean(Xt_RK, 1);
Xt_EM_mean = mean(Xt_EM, 1);
t = ti : dt : tf;

figure, hold on;
plot(t, Xt_RK_mean(1 : end - 1)), plot(t, Xt_EM_mean(1 : end - 1));
grid minor, title('Euler-Maruyama vs SSPRK3');