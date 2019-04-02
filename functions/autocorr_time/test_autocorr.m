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

%% Var autocorr time test
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
    
%     actime(k) = initial_positive_estimator(c, bt_test) / (N / durationT);
    actime(k) = autocorr_time(c, bt_test) / (N / durationT);
end

[max_error, max_pos] = max(abs(actime' / sqrt(2 * pi) - tau))
[min_error, min_pos] = min(abs(actime' / sqrt(2 * pi) - tau))
error = abs(actime' / sqrt(2 * pi) - tau) / durationT;

% Plot the estimation error
figure, plot(tau, error);
title('Estimation error'), grid minor;

%% var delta t
clear all, close all, clc;

N = 10 : 1000;
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
    
%     actime(k) = initial_positive_estimator(c, bt_test) / (N(k) / durationT);
    actime(k) = autocorr_time(c, bt_test) / (N(k) / durationT);
end

[max_error, max_pos] = max(abs(actime' / sqrt(2 * pi) - tau))
[min_error, min_pos] = min(abs(actime' / sqrt(2 * pi) - tau))
error = abs(actime' / sqrt(2 * pi) - tau) / durationT;

% Plot of the estimated tau as a function of N
figure, hold on;
plot(actime / sqrt(2 * pi)), plot(tau * ones(size(actime)));
xlabel('N')
title('Autocorrelation time estimation'), grid minor;

% Plot of the estimation error
figure, plot(N, error);
xlabel('N'), grid minor;
title('Normalized estimation error');

% Log plot of error
figure, semilogx(N, log(error));
xlabel('N'), grid minor;
title('Error logplot');

%% heterogenous test
clear all, close all, clc;

alin = linspace(0.1, 3.0, 10);
T = 5.0;
N = 1000;
t_0 = 0.0;
t_f = 100.0;
durationT = t_f - t_0;
deltaT = (durationT / N);
tau = 20.0;
sigma = 10.0;

beta = @(x, alin) 1 + alin * cos(2 * pi * 3 * x / T);
alpha = @(x, alin) 1 + alin * sin(2 * pi * 2 * x / T);
cov_x = @(tau, t1, t2) sigma.^2 * ...
    exp(-0.5 * ((t1 - t2) / tau ).^2);

bt_test = zeros(N, 1);

actime = zeros(length(alin), 1);
for k = 1 : length(alin)    
    c = zeros(N);
    for i = 1 : N
        for j = 1 : N
            t1 = i / beta((i + j) / 2, alin);
            t2 = j / beta((i + j) / 2, alin);
            c(i, j) = cov_x(tau, t1, t2, alin(k)) * alpha((t1 + t2) / 2, alin);
        end
    end
    
%     actime(k) = initial_positive_estimator(c, bt_test) / (N(k) / durationT);
    actime(k) = autocorr_time(c, bt_test) / (N / durationT);
end

[max_error, max_pos] = max(abs(actime' / sqrt(2 * pi) - tau))
[min_error, min_pos] = min(abs(actime' / sqrt(2 * pi) - tau))
error = abs(actime' / sqrt(2 * pi) - tau) / durationT;

% Plot of the expected value and its estimation
figure, hold on;
plot(actime / sqrt(2 * pi)), plot(tau * ones(size(actime)));
xlabel('Non-linear coefficient')
title('Autocorrelation time estimation'), grid minor;