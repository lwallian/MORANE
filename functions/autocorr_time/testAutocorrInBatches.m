%
% Author: Agustin PICARD, intern @ Scalian with Valentin RESSEGUIER as
% supervisor
%

%% Test autocorrelation time in batches with real data
clear all, close all, clc;

data = load('C_DNS100_2Modes.mat');
% data = load('C_DNS300_2Modes.mat');
cov_v = data.c;
bt = data.bt;
clear data;

autocorrTime = autocorrelationTimeInBatches(cov_v, bt);

figure, plot(autocorrTime), grid minor;
title('Autocorrelation time (DNS100 - 8 modes)');
xlabel('Period'), ylabel('\tau_{corr}');

%% Test autocorrelation time in batches with synthetic data
clear all, close all, clc;

T = 5.0;
N = 1000;
t_0 = 0.0;
t_f = 100.0;
durationT = t_f - t_0;
deltaT = (durationT / N);
t = linspace(t_0, t_f, N);
tau = 2.0;
sigma = 1 / (sqrt(2 * pi) * (tau / deltaT));

beta = @(x) 1 + 1 * cos(2 * pi * 3 * x / T);
alpha = @(x) 1 + 1 * sin(2 * pi * 2 * x / T);

var_alin = alpha(t ./ 2);
var_alin = var_alin ./ mean(var_alin);

tau_alin = beta(t ./ 2);
tau_alin = tau_alin ./ mean(tau_alin);

cov_x = sigma.^2 .* var_alin .* exp(-0.5 * (t ./ (tau_alin .* tau)).^2);

c = zeros(N);
for i = 1 : N
    for j = 1 : N
        c(i, j) = cov_x(abs(i - j) + 1);
    end
end

bt_test = zeros(N, 1); % Disregard the computation of the large-scale velocity

actime = autocorrelationTimeInBatches(c, bt_test) / (N / durationT);

figure, plot(actime), title('Autocorrelation time (synthetic)');
xlabel('Period'), ylabel('\tau_{corr}');
