%
% Author: Agustin PICARD, intern @ Scalian with Valentin RESSEGUIER as
% supervisor
%
clear all, close all, clc;

data = load('C_DNS100_8Modes.mat');
dt = 0.05;
% data = load('C_DNS300_16Modes.mat');
% dt = 0.25;
cov_v = data.c;
cov_v = cov_v(1 : 10000, 1 : 10000);
bt = data.bt;
clear data;
cov_s = smallScaleVelocityCov(cov_v, bt);
correlation = estimateAutocorrelation(cov_s);
period = periodicityFromAutocorrelation(cov_v);

%% Test LMS / NLMS filter to clean the periodic part
% Variation of delay and number of taps
maxDelay = 5 * period;
maxTaps = 1000;
tau = zeros(maxDelay, maxTaps);
for d = 1 : maxDelay
    for l = 1 : maxTaps
        filteredCorrelation = LMSFilter(correlation, l, d);
%         filteredCorrelation = RLSFilter(correlation, l, d, 0.95);
        tau(d, l) = 1 + 2 * sum(filteredCorrelation);
    end
end

figure, imagesc(tau), axis xy;
title('Variation of LMS filter parameters');
ylabel('Delay'), xlabel('Filter order');

% Plot the results for some fixed delays and orders
figure, plot(tau(period, :)), grid minor, title('Filter order variation for D = T_{per}'), xlabel('Filter order'), ylabel('$\tau_{corr} / \Delta t$', 'Interpreter', 'latex');
figure, plot(tau(2 * period, :)), grid minor, title('Filter order variation for D = 2 T_{per}'), xlabel('Filter order'), ylabel('$\tau_{corr} / \Delta t$', 'Interpreter', 'latex');
figure, plot(tau(floor(1.5 * period), :)), grid minor, title('Filter order variation for D = 1.5 T_{per}'), xlabel('Filter order'), ylabel('$\tau_{corr} / \Delta t$', 'Interpreter', 'latex');
figure, plot(tau(:, 100)), grid minor;
figure, plot(tau(:, 400)), grid minor;

%% RLS variation with order and forget factor
d = period;
maxTaps = 100;
forgetPoints = 1000;
forget = linspace(0.9, 0.9999, forgetPoints);

tau = zeros(maxTaps, forgetPoints);
tau_cut = zeros(maxTaps, forgetPoints);
for f = 1 : forgetPoints
    for l = 1 : maxTaps
        filteredCorrelation = RLSFilter(correlation, l, d, forget(f));
        tau(f, l) = 1 + 2 * sum(filteredCorrelation);
        tau_cut(f, l) = 1 + 2 * sum(filteredCorrelation(1 : 5 * period));
    end
end

figure, imagesc(tau), axis xy;
title('Variation of LMS filter parameters');
ylabel('Forget factor'), xlabel('Filter order');

figure, imagesc(tau_cut), axis xy;
title('Variation of LMS filter parameters');
ylabel('Forget factor'), xlabel('Filter order');

%% Visualization of the variability of the filter's parameters

d = 2 * period;
l = 0.5 * d + 1;
figure, subplot(4, 1, 1);
plot(correlation), title('Correlation function'), grid minor;
filteredCorrelation = LMSFilter(correlation, l, d);
tau_corr = 1 + 2 * sum(filteredCorrelation);
subplot(4, 1, 2), plot(filteredCorrelation), title(['Filtered with D = ' num2str(d), ' order = ' num2str(l) ' \tau = ' num2str(tau_corr)]), grid minor;
filteredCorrelation = LMSFilter(correlation, l + 1, d);
tau_corr = 1 + 2 * sum(filteredCorrelation);
subplot(4, 1, 3), plot(filteredCorrelation), grid minor;
title(['Filtered with D = ' num2str(d), ' order = ' num2str(l + 1) ' \tau = ' num2str(tau_corr)]);
filteredCorrelation = LMSFilter(correlation, l + 2, d);
tau_corr = 1 + 2 * sum(filteredCorrelation);
subplot(4, 1, 4), plot(filteredCorrelation), grid minor; 
title(['Filtered with D = ' num2str(d), ' order = ' num2str(l + 2) ' \tau = ' num2str(tau_corr)]);

%% Visualization of the variability with the order
d = period;
maxTaps = 200;
forgetFactor = 0.999;

tau = zeros(maxTaps, 1);
tau_cut = zeros(maxTaps, 1);
for l = 1 : maxTaps
    filteredCorrelation = LMSFilter(correlation, l, d);
%     filteredCorrelation = RLSFilter(correlation, l, d, forgetFactor);
    tau(l) = 1 + 2 * sum(filteredCorrelation);
    tau_cut(l) = 1 + 2 * sum(filteredCorrelation(1 : 5 * period));
end

% Plot the correlation time estimations
figure, hold on;
% plot(tau), plot(tau_cut), grid minor;
plot(tau .* dt), plot(tau_cut .* dt), grid minor;
title('$\hat{\tau}_{corr}$ estimation for different filter orders', 'Interpreter', 'latex');
xlabel('Filter order'), ylabel('$\tau_{corr} [s]$', 'Interpreter', 'latex');

%% Search of the optimal filter order
d = 1 * period;
maxTaps = 200;
forgetFactor = 0.999;

filteredVariation = zeros(maxTaps, 1);
for i = 1 : maxTaps
%     filteredCorrelation = LMSFilter(correlation, i, d);
    filteredCorrelation = RLSFilter(correlation, l, d, forgetFactor);
    filteredVariation(i) = norm(diff(filteredCorrelation(1 : 5 * period)));
end

figure, plot(filteredVariation), grid minor;
title('Norm of the gradient for different filter orders')
xlabel('Filter order')
ylabel('$||\partial_t \hat{Cov_s^\epsilon}||_2^2$', 'Interpreter', 'latex');

%% Normal cut sweep
N = size(cov_v, 2);
var_s = trace(cov_s);

sweep = zeros(N - 1, 1);
aperiodic = mean(diag(cov_s, 1));
sweep(1) =  1 + 2 * aperiodic ./ var_s;
for i = 2 : N
%     aperiodic = [aperiodic ./ (i - 1), mean(diag(cov_s, i))] .* i;
    aperiodic = [aperiodic ./ N, mean(diag(cov_s, i))] .* N;
    sweep(i - 1) = 1 + 2 * sum(aperiodic) / var_s;
end

% figure, plot(0 : dt : dt * (length(sweep) - 1), sweep);
figure, plot(0 : dt : dt * (length(sweep) - 1), sweep * dt);
% figure, plot(linspace(0.0, 10000 * 0.05, length(tau)), tau * 0.05);
title('Sweep (DNS100 - 2 modes)')
xlabel('Time [s]'), ylabel('$\tau_{corr} [s]$', 'Interpreter', 'latex');
grid minor;

tau_cut = sweep(5 * period) * dt

%% Aperiodic cut sweep

tau = aperiodicCorrelationTimeSweep(cov_v, bt);
% tau = aperiodicCorrelationTimeSweep(cov_v(1 : 10000, 1 : 10000), bt);

% figure, plot(tau);
figure, plot(0 : dt : dt * (length(tau) - 1), tau * dt);
title('Aperiodic correlation time estimation (DNS300 - 2 modes)')
xlabel('Time [s]'), ylabel('$\tau_{corr} [s]$', 'Interpreter', 'latex');
grid minor;

%% LMS cut sweep

d = 1 * period;
taps = 79;
forgetFactor = 0.999;

% filteredCorrelation = LMSFilter(correlation, taps, d);
% filteredCorrelation = RLSFilter(correlation, taps, d, forgetFactor);
[filteredCorrelation, taps] = minimalVarianceLMS(correlation, 500, d);
sweep = 1 + 2 * cumsum(filteredCorrelation);

figure, plot(0 : dt : dt * (length(sweep) - 1), sweep), grid minor;
title(['LMS filtered \tau_{corr} estimation sweep (taps = ', num2str(taps) ')'])
xlabel('Time [s]'), ylabel('$\frac{\tau_{corr}}{\Delta t}$', 'Interpreter', 'latex');

%% Simple estimator

tau_corr = sqrt(2 * mean((correlation).^2) / mean((diff(correlation) ./ dt).^2))
tau_corr_cut = sqrt(2 * mean(correlation(1 : (5 * period)).^2) / mean(diff((correlation(1 : (5 * period))) ./ dt).^2))

%% Filtered - Normal - Oscillating comparison

d = 1 * period;
[optimalFiltered, optimalTaps] = minimalVarianceLMS(correlation, 500, d);
filteredCorrelation = LMSFilter(correlation, 150, d);

% Plot first the unfiltered correlation and then the other two
figure;
subplot(3, 1, 1), plot(0 : dt : dt * (length(correlation) - 1), correlation);
title('Unfiltered'), grid minor, ylabel('Cov_s');
subplot(3, 1, 2), plot(0 : dt : dt * (length(optimalFiltered) - 1), optimalFiltered);
title('Optimal filtering'), grid minor, ylabel('Filtered Cov_s');
subplot(3, 1, 3), plot(0 : dt : dt * (length(filteredCorrelation) - 1), filteredCorrelation);
title('Oscillating filtering'), grid minor, ylabel('Filtered Cov_s');
xlabel('Time [s]');

%% Minimal variance test

d = 1 * period;
maxLag = 5 * period;

[filteredCorrelation, taps] = minimalVarianceLMS(correlation, 500, d);
tau = (1 + 2 * sum(filteredCorrelation)) * dt
tau_cut = (1 + 2 * sum(filteredCorrelation(1 : maxLag))) * dt

%% Separation of periodic and aperiodic parts

d = 1 * period;
maxLag = 5 * period;

[filteredCorrelation, taps] = minimalVarianceLMS(correlation, 500, d);
refSignal = correlation;
N = length(refSignal);

% Initialize filter
h = zeros(taps, 1);
inputSignal = [zeros(d, 1)', refSignal'];
inputSignal = inputSignal(1 : N);

outSignal = zeros(N, 1);
periodic = zeros(N, 1);
for i = 1 : N - taps - 1
    inputSlice = inputSignal(i : i + taps - 1)';
    outSignal(i) = refSignal(i) - h' * inputSlice;
    h = h + (2 / (norm(inputSlice).^2 + 1e-4 * taps * refSignal(1).^2)) * outSignal(i) * inputSlice;
    periodic(i) = h' * inputSlice; % for test
%     h = h + 0.01 * outSignal(i) * inputSlice; % another step size
end

figure;
subplot(3, 1, 1), plot(0 : dt : (length(correlation) - 1) * dt, correlation);
title('Original signal'), grid minor;
subplot(3, 1, 2), plot(0 : dt : (length(outSignal) - 1) * dt, outSignal);
title('Aperiodic part'), grid minor;
subplot(3, 1, 3), plot(0 : dt : (length(periodic) - 1) * dt, periodic);
title('Periodic part'), grid minor;
