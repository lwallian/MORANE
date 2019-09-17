%
% Author: Agustin PICARD, intern @ Scalian with Valentin RESSEGUIER as
% supervisor
%
% First tests of the correlation estimator
%% Test with real data
clear all, close all, clc;

% data = load('C_DNS100_2Modes.mat');
data = load('C_DNS300_2Modes.mat');
cov_v = data.c;
bt = data.bt;
clear data;

autocorr_time(cov_v, bt)

%% Synthetic test (simple)
clear all, close all, clc;

N = [374, 374];
% cov_x = ones(N);
cov_x = eye(N);
bt_test = zeros(N(1), 1);
is_big_data = true;

autocorr_time(is_big_data, cov_x, bt_test)

%% Gaussian test
clear all, close all, clc;

T = 5.0;
N = 1000;
t_0 = 0.0;
t_f = 100.0;
durationT = t_f - t_0;
deltaT = (durationT / N);
t = linspace(t_0, t_f, N);
tau = 50.0;
sigma = 1 / (sqrt(2 * pi) * (tau / deltaT));

beta = @(x) 1 + 0.2 * cos(2 * pi * 3 * x / T);
alpha = @(x) 1 + 0.2 * sin(2 * pi * 2 * x / T);

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

figure, imagesc(c), title('covariance matrix');
figure, plot(cov_x), title('cov reel');

autocorr_time(c, bt_test) / (N / durationT)

%% Variation of the autocorr time with different parameters
clear all, close all, clc;

N = 1000;
t_0 = 0.0;
t_f = 100.0;
durationT = t_f - t_0;
t = linspace(t_0, t_f, N);
tau = 0.1 : 0.1 : t_f;
sigma = 10.0;

actime = zeros(length(tau), 1);
for k = 1 : length(tau)
    cov_x = sigma.^2 .* exp(-0.5 * (t ./ tau(k)).^2);
    
    c = zeros(N);
    for i = 1 : N
        for j = 1 : N
            c(i, j) = cov_x(abs(i - j) + 1);
        end
    end
    
    bt_test = zeros(N, 1);
    
    actime(k) = autocorr_time(c, bt_test) / (N / durationT);
end

[max_error, max_pos] = max(abs(actime' / sqrt(2 * pi) - tau))
[min_error, min_pos] = min(abs(actime' / sqrt(2 * pi) - tau))
error = sqrt(actime'.^2 / (2 * pi) - tau.^2) / durationT;

% Plot the estimation error
figure, plot(tau, abs(error));
title('Estimator bias'), grid minor;
xlabel('\tau [s]'), ylabel('$||\tau - \hat{\tau}||_2^2$ [s]', 'Interpreter', 'latex');

%% Variation of the correlation time estimator with delta t
clear all, close all, clc;

N = 10 : 100 : 10000;
t_0 = 0.0;
t_f = 100.0;
durationT = t_f - t_0;
tau = 2.0;
sigma = 10.0;

actime = zeros(length(N), 1);
for k = 1 : length(N)
    t = linspace(t_0, t_f, N(k));
    cov_x = sigma.^2 * exp(-0.5 * (t ./ tau).^2);
    
    c = zeros(N(k));
    for i = 1 : N(k)
        for j = 1 : N(k)
            c(i, j) = cov_x(abs(i - j) + 1);
        end
    end
    
    bt_test = zeros(N(k), 1);
    
%     actime(k) = autocorr_time(c, bt_test) / (N(k) / durationT);
    actime(k) = autocorrelationTimeInBatches(c, bt_test, 'global') / (N(k) / durationT);
end

[max_error, max_pos] = max(abs(actime' / sqrt(2 * pi) - tau))
[min_error, min_pos] = min(abs(actime' / sqrt(2 * pi) - tau))
error = sqrt(actime'.^2 / (2 * pi) - tau.^2) / durationT;

% Plot of the estimated tau as a function of N
figure, hold on;
plot(actime / sqrt(2 * pi)), plot(tau * ones(size(actime)));
xlabel('N [samples]'), ylabel('$\hat{\tau} [s]$', 'Interpreter', 'latex')
title('Autocorrelation time estimation'), grid minor;

% Plot of the estimation error
figure, plot(N, abs(error));
xlabel('N [samples]'), grid minor;
ylabel('$||\tau - \hat{\tau}||_2^2$ [s]', 'Interpreter', 'latex');
title('Normalized estimation error');

% Log plot of error
figure, semilogx(N, log10(abs(error)));
xlabel('N [samples]'), grid minor;
ylabel('$\log\left(||\tau - \hat{\tau}||_2^2\right)$ [s]', 'Interpreter', 'latex');
title('Error logplot');

%% Test with heterogeneous correlation matrix to test for robustness (amplitude)
clear all, close all, clc;

alin = linspace(0.1, 3.0, 100);
T = 5.0;
N = 1000;
t_0 = 0.0;
t_f = 100.0;
durationT = t_f - t_0;
deltaT = (durationT / N);
tau = 20.0;
sigma = 10.0;

beta = @(x, alin) 1 + alin * cos(2 * pi * x / T);
alpha = @(x, alin) 1 + alin * sin(2 * pi * x / T);
cov_x = @(tau, dt, st, alin) sigma.^2 * alpha(st, alin) * ...
    exp(-0.5 * (dt / tau / beta(st, alin) ).^2);

bt_test = zeros(N, 1);

actime = zeros(length(alin), 1);
for k = 1 : length(alin)
    k
    c = zeros(N);
    for i = 1 : N
        for j = 1 : N
            dt = abs(i - j) * deltaT;
            st = (i + j) * deltaT;
            c(i, j) = cov_x(tau, dt, st, alin(k));
        end
    end
    
%     actime(k) = initial_positive_estimator(c, bt_test) / (N(k) / durationT);
    actime(k) = autocorr_time(c, bt_test) / (N / durationT);
end

[max_error, max_pos] = max(abs(actime' / sqrt(2 * pi) - tau))
[min_error, min_pos] = min(abs(actime' / sqrt(2 * pi) - tau))
error = sqrt(actime'.^2 / (2 * pi) - tau.^2) / durationT;

% Plot of the expected value and its estimation
figure, hold on;
plot(alin, actime / sqrt(2 * pi)), plot(alin, tau * ones(size(actime)));
xlabel('Non-linearity modulation amplitude [$\frac{m^2}{s^2}$]', 'Interpreter', 'latex')
ylabel('\tau [s]')
title('Autocorrelation time estimation'), grid minor;

% Plot of the incurred error
figure, plot(alin, abs(error));
xlabel('Non-linearity modulation amplitude [$\frac{m^2}{s^2}$]', 'Interpreter', 'latex')
ylabel('$||\tau - \hat{\tau}||_2^2$ [s]', 'Interpreter', 'latex');
title('Normalized error'), grid minor;

%% Test with heterogeneous correlation matrix to test for robustness (period)
clear all, close all, clc;

alin = 0.5;
T = 1 : 100;
N = 1000;
t_0 = 0.0;
t_f = 100.0;
durationT = t_f - t_0;
deltaT = (durationT / N);
tau = 20.0;
sigma = 10.0;

beta = @(x, T) 1 + alin * cos(2 * pi * x / T);
alpha = @(x, T) 1 + alin * sin(2 * pi * x / T);
cov_x = @(tau, dt, st, T) sigma.^2 * alpha(st, T) * ...
    exp(-0.5 * (dt / tau / beta(st, T) ).^2);

bt_test = zeros(N, 1);

actime = zeros(length(T), 1);
for k = 1 : length(T)
    k
    c = zeros(N);
    for i = 1 : N
        for j = 1 : N
            dt = abs(i - j) * deltaT;
            st = (i + j) * deltaT;
            c(i, j) = cov_x(tau, dt, st, T(k));
        end
    end
    
%     actime(k) = initial_positive_estimator(c, bt_test) / (N(k) / durationT);
    actime(k) = autocorr_time(c, bt_test) / (N / durationT);
end

[max_error, max_pos] = max(abs(actime' / sqrt(2 * pi) - tau))
[min_error, min_pos] = min(abs(actime' / sqrt(2 * pi) - tau))
error = sqrt(actime'.^2 / (2 * pi) - tau.^2) / durationT;

% Plot of the expected value and its estimation
figure, hold on;
plot(T, actime / sqrt(2 * pi)), plot(T, tau * ones(size(actime)));
xlabel('Non-linearity modulation period [s]'), ylabel('\tau [s]')
title('Autocorrelation time estimation'), grid minor;

% Plot of the incurred error
figure, plot(T, abs(error));
xlabel('Non-linearity modulation period [s]')
ylabel('$||\tau - \hat{\tau}||_2^2$ [s]', 'Interpreter', 'latex');
title('Normalized error'), grid minor;
