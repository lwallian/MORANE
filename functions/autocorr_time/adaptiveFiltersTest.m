%
% Author: Agustin PICARD, intern @ Scalian with Valentin RESSEGUIER as
% supervisor
%
clear all, close all, clc;

data = load('C_DNS100_8Modes.mat');
% data = load('C_DNS300_2Modes.mat');
cov_v = data.c;
bt = data.bt;
clear data;
cov_s = smallScaleVelocityCov(cov_v, bt);
correlation = estimateAutocorrelation(cov_s);
correlation = correlation(1 : 10000);
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
maxTaps = 1000;

tau = zeros(maxTaps, 1);
tau_cut = zeros(maxTaps, 1);
for l = 1 : maxTaps
    filteredCorrelation = LMSFilter(correlation, l, d);
    tau(l) = 1 + 2 * sum(filteredCorrelation);
    tau_cut(l) = 1 + 2 * sum(filteredCorrelation(1 : 1000));
end

% Plot the correlation time estimations
figure, hold on;
plot(tau), plot(tau_cut), grid minor;
title('$\tau_{corr}$ estimation for different filter orders', 'Interpreter', 'latex');
xlabel('Filter order'), ylabel('$\frac{\tau_{corr}}{\Delta t}$', 'Interpreter', 'latex');

%% Search of the optimal filter order
d = 3 * period;
maxTaps = 1000;
N = length(correlation);

filteredVariation = zeros(maxTaps);
for i = 1 : maxTaps
    filteredCorrelation = LMSFilter(correlation, i, d);
    filteredVariation(i) = norm(diff(filteredCorrelation(1 : 500)));
end

figure, plot(filteredVariation), grid minor;
