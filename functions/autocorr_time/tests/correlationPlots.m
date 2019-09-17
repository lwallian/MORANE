%% Just plots the correlation functions and their respective downsampled versions
clear all, close all, clc;

%% DNS100
data = load('C_DNS100_2Modes.mat');
dt = 0.05;
cov_v = data.c;
cov_v = cov_v(1 : 10000, 1 : 10000);
bt = data.bt;
clear data;
cov_s = smallScaleVelocityCov(cov_v, bt);
correlation = estimateAutocorrelation(cov_s);
power_spectrum = fftshift(fft([correlation', zeros(size(correlation'))]));
power_spectrum = abs(power_spectrum(floor(length(power_spectrum) / 2) : end));
freq_axis = linspace(0.0, 1 / (2 * dt), length(power_spectrum));

figure, subplot(2, 1, 1);
plot(0 : dt : dt * (length(correlation) - 1), correlation);
title('Unresolved velocity correlation DNS100 2 Modes');
xlabel('Time [s]'), grid minor;
subplot(2, 1, 2);
plot(freq_axis, power_spectrum);
title('Unresolved velocity power Spectrum DNS100 2 Modes')
xlabel('Frequency [Hz]'), grid minor;
print('C:\Users\agustinmartin.picard\Documents\Correlation Functions\DNS100_2modes.png', '-dpng', '-r500');

data = load('C_DNS100_4Modes.mat');
dt = 0.05;
cov_v = data.c;
cov_v = cov_v(1 : 10000, 1 : 10000);
bt = data.bt;
clear data;
cov_s = smallScaleVelocityCov(cov_v, bt);
correlation = estimateAutocorrelation(cov_s);
power_spectrum = fftshift(fft([correlation', zeros(size(correlation'))]));
power_spectrum = abs(power_spectrum(floor(length(power_spectrum) / 2) : end));

figure, subplot(2, 1, 1);
plot(0 : dt : dt * (length(correlation) - 1), correlation);
title('Unresolved velocity correlation DNS100 4 Modes');
xlabel('Time [s]'), grid minor;
subplot(2, 1, 2);
plot(freq_axis, power_spectrum);
title('Unresolved velocity power Spectrum DNS100 4 Modes')
xlabel('Frequency [Hz]'), grid minor;
print('C:\Users\agustinmartin.picard\Documents\Correlation Functions\DNS100_4modes.png', '-dpng', '-r500');

data = load('C_DNS100_8Modes.mat');
dt = 0.05;
cov_v = data.c;
cov_v = cov_v(1 : 10000, 1 : 10000);
bt = data.bt;
clear data;
cov_s = smallScaleVelocityCov(cov_v, bt);
correlation = estimateAutocorrelation(cov_s);
power_spectrum = fftshift(fft([correlation', zeros(size(correlation'))]));
power_spectrum = abs(power_spectrum(floor(length(power_spectrum) / 2) : end));

figure, subplot(2, 1, 1);
plot(0 : dt : dt * (length(correlation) - 1), correlation);
title('Unresolved velocity correlation DNS100 8 Modes');
xlabel('Time [s]'), grid minor;
subplot(2, 1, 2);
plot(freq_axis, power_spectrum);
title('Unresolved velocity power Spectrum DNS100 8 Modes')
xlabel('Frequency [Hz]'), grid minor;
print('C:\Users\agustinmartin.picard\Documents\Correlation Functions\DNS100_8modes.png', '-dpng', '-r500');

data = load('C_DNS100_16Modes.mat');
dt = 0.05;
cov_v = data.c;
cov_v = cov_v(1 : 10000, 1 : 10000);
bt = data.bt;
clear data;
cov_s = smallScaleVelocityCov(cov_v, bt);
correlation = estimateAutocorrelation(cov_s);
power_spectrum = fftshift(fft([correlation', zeros(size(correlation'))]));
power_spectrum = abs(power_spectrum(floor(length(power_spectrum) / 2) : end));

figure, subplot(2, 1, 1);
plot(0 : dt : dt * (length(correlation) - 1), correlation);
title('Unresolved velocity correlation DNS100 16 Modes');
xlabel('Time [s]'), grid minor;
subplot(2, 1, 2);
plot(freq_axis, power_spectrum);
title('Unresolved velocity power Spectrum DNS100 16 Modes')
xlabel('Frequency [Hz]'), grid minor;
print('C:\Users\agustinmartin.picard\Documents\Correlation Functions\DNS100_16modes.png', '-dpng', '-r500');

%% DNS300
data = load('C_DNS300_2Modes.mat');
dt = 0.25;
cov_v = data.c;
bt = data.bt;
clear data;
cov_s = smallScaleVelocityCov(cov_v, bt);
correlation = estimateAutocorrelation(cov_s);
power_spectrum = fftshift(fft([correlation', zeros(size(correlation'))]));
power_spectrum = abs(power_spectrum(floor(length(power_spectrum) / 2) : end));
freq_axis = linspace(0.0, 1 / (2 * dt), length(power_spectrum));

figure, subplot(2, 1, 1);
plot(0 : dt : dt * (length(correlation) - 1), correlation);
title('Unresolved velocity correlation DNS300 2 Modes');
xlabel('Time [s]'), grid minor;
subplot(2, 1, 2);
plot(freq_axis, power_spectrum);
title('Unresolved velocity power Spectrum DNS300 2 Modes')
xlabel('Frequency [Hz]'), grid minor;
print('C:\Users\agustinmartin.picard\Documents\Correlation Functions\DNS300_2modes.png', '-dpng', '-r500');

data = load('C_DNS300_4Modes.mat');
dt = 0.25;
cov_v = data.c;
bt = data.bt;
clear data;
cov_s = smallScaleVelocityCov(cov_v, bt);
correlation = estimateAutocorrelation(cov_s);
power_spectrum = fftshift(fft([correlation', zeros(size(correlation'))]));
power_spectrum = abs(power_spectrum(floor(length(power_spectrum) / 2) : end));

figure, subplot(2, 1, 1);
plot(0 : dt : dt * (length(correlation) - 1), correlation);
title('Unresolved velocity correlation DNS300 4 Modes');
xlabel('Time [s]'), grid minor;
subplot(2, 1, 2);
plot(freq_axis, power_spectrum);
title('Unresolved velocity power Spectrum DNS300 4 Modes')
xlabel('Frequency [Hz]'), grid minor;
print('C:\Users\agustinmartin.picard\Documents\Correlation Functions\DNS300_4modes.png', '-dpng', '-r500');

data = load('C_DNS300_8Modes.mat');
dt = 0.25;
cov_v = data.c;
bt = data.bt;
clear data;
cov_s = smallScaleVelocityCov(cov_v, bt);
correlation = estimateAutocorrelation(cov_s);
power_spectrum = fftshift(fft([correlation', zeros(size(correlation'))]));
power_spectrum = abs(power_spectrum(floor(length(power_spectrum) / 2) : end));

figure, subplot(2, 1, 1);
plot(0 : dt : dt * (length(correlation) - 1), correlation);
title('Unresolved velocity correlation DNS300 8 Modes');
xlabel('Time [s]'), grid minor;
subplot(2, 1, 2);
plot(freq_axis, power_spectrum);
title('Unresolved velocity power Spectrum DNS300 8 Modes')
xlabel('Frequency [Hz]'), grid minor;
print('C:\Users\agustinmartin.picard\Documents\Correlation Functions\DNS300_8modes.png', '-dpng', '-r500');

data = load('C_DNS300_16Modes.mat');
dt = 0.25;
cov_v = data.c;
bt = data.bt;
clear data;
cov_s = smallScaleVelocityCov(cov_v, bt);
correlation = estimateAutocorrelation(cov_s);
power_spectrum = fftshift(fft([correlation', zeros(size(correlation'))]));
power_spectrum = abs(power_spectrum(floor(length(power_spectrum) / 2) : end));

figure, subplot(2, 1, 1);
plot(0 : dt : dt * (length(correlation) - 1), correlation);
title('Unresolved velocity correlation DNS300 16 Modes');
xlabel('Time [s]'), grid minor;
subplot(2, 1, 2);
plot(freq_axis, power_spectrum);
title('Unresolved velocity power Spectrum DNS300 16 Modes')
xlabel('Frequency [Hz]'), grid minor;
print('C:\Users\agustinmartin.picard\Documents\Correlation Functions\DNS300_16modes.png', '-dpng', '-r500');

%% Downsampled version: LMS estimated rate
% DNS100
clear all;
data = load('C_DNS100_2Modes.mat');
dt = 0.05;
cov_v = data.c;
cov_v = cov_v(1 : 10000, 1 : 10000);
bt = data.bt;
clear data;

% Estimate the downsampling rate
tau = correlationTimeLMS(cov_v, bt, dt);
tau = max(floor(tau), 1);

% Downsample by tau
% cov_v = cov_v(1 : tau : end, 1 : tau : end);
% bt = bt(1 : tau : end, :);

% Estimate the correlation function and its spectrum
cov_s = smallScaleVelocityCov(cov_v, bt);
cov_s = cov_s(1 : tau : end, 1 : tau : end);
correlation = estimateAutocorrelation(cov_s);
power_spectrum = fftshift(fft([correlation', zeros(size(correlation'))]));
power_spectrum = abs(power_spectrum(floor(length(power_spectrum) / 2) : end));
freq_axis = linspace(0.0, 1 / (2 * dt), length(power_spectrum));

figure, subplot(2, 1, 1);
plot(0 : dt : dt * (length(correlation) - 1), correlation);
title(['Downsampled unresolved velocity correlation DNS100 2 Modes (\tau =', num2str(tau), ')']);
xlabel('Time [s]'), grid minor;
subplot(2, 1, 2);
plot(freq_axis, power_spectrum), title('Power Spectrum')
xlabel('Frequency [Hz]'), grid minor;
print('C:\Users\agustinmartin.picard\Documents\Correlation Functions\DNS100_2modes_downsampled_LMS.png', '-dpng', '-r500');


data = load('C_DNS100_4Modes.mat');
dt = 0.05;
cov_v = data.c;
cov_v = cov_v(1 : 10000, 1 : 10000);
bt = data.bt;
clear data;

% Estimate the downsampling rate
tau = correlationTimeLMS(cov_v, bt, dt);
tau = max(floor(tau), 1);

% Downsample by tau
% cov_v = cov_v(1 : tau : end, 1 : tau : end);
% bt = bt(1 : tau : end, :);

% Estimate the correlation function and its spectrum
cov_s = smallScaleVelocityCov(cov_v, bt);
cov_s = cov_s(1 : tau : end, 1 : tau : end);
correlation = estimateAutocorrelation(cov_s);
power_spectrum = fftshift(fft([correlation', zeros(size(correlation'))]));
power_spectrum = abs(power_spectrum(floor(length(power_spectrum) / 2) : end));
freq_axis = linspace(0.0, 1 / (2 * dt), length(power_spectrum));

figure, subplot(2, 1, 1);
plot(0 : dt : dt * (length(correlation) - 1), correlation);
title(['Downsampled unresolved velocity correlation DNS100 4 Modes (\tau =', num2str(tau), ')']);
xlabel('Time [s]'), grid minor;
subplot(2, 1, 2);
plot(freq_axis, power_spectrum), title('Power Spectrum')
xlabel('Frequency [Hz]'), grid minor;
print('C:\Users\agustinmartin.picard\Documents\Correlation Functions\DNS100_4modes_downsampled_LMS.png', '-dpng', '-r500');


data = load('C_DNS100_8Modes.mat');
dt = 0.05;
cov_v = data.c;
cov_v = cov_v(1 : 10000, 1 : 10000);
bt = data.bt;
clear data;

% Estimate the downsampling rate
tau = correlationTimeLMS(cov_v, bt, dt);
tau = max(floor(tau), 1);

% Downsample by tau
% cov_v = cov_v(1 : tau : end, 1 : tau : end);
% bt = bt(1 : tau : end, :);

% Estimate the correlation function and its spectrum
cov_s = smallScaleVelocityCov(cov_v, bt);
cov_s = cov_s(1 : tau : end, 1 : tau : end);
correlation = estimateAutocorrelation(cov_s);
power_spectrum = fftshift(fft([correlation', zeros(size(correlation'))]));
power_spectrum = abs(power_spectrum(floor(length(power_spectrum) / 2) : end));
freq_axis = linspace(0.0, 1 / (2 * dt), length(power_spectrum));

figure, subplot(2, 1, 1);
plot(0 : dt : dt * (length(correlation) - 1), correlation);
title(['Downsampled unresolved velocity correlation DNS100 8 Modes (\tau =', num2str(tau), ')']);
xlabel('Time [s]'), grid minor;
subplot(2, 1, 2);
plot(freq_axis, power_spectrum), title('Power Spectrum')
xlabel('Frequency [Hz]'), grid minor;
print('C:\Users\agustinmartin.picard\Documents\Correlation Functions\DNS100_8modes_downsampled_LMS.png', '-dpng', '-r500');


data = load('C_DNS100_16Modes.mat');
dt = 0.05;
cov_v = data.c;
cov_v = cov_v(1 : 10000, 1 : 10000);
bt = data.bt;
clear data;

% Estimate the downsampling rate
tau = correlationTimeLMS(cov_v, bt, dt);
tau = max(floor(tau), 1);

% Downsample by tau
% cov_v = cov_v(1 : tau : end, 1 : tau : end);
% bt = bt(1 : tau : end, :);

% Estimate the correlation function and its spectrum
cov_s = smallScaleVelocityCov(cov_v, bt);
cov_s = cov_s(1 : tau : end, 1 : tau : end);
correlation = estimateAutocorrelation(cov_s);
power_spectrum = fftshift(fft([correlation', zeros(size(correlation'))]));
power_spectrum = abs(power_spectrum(floor(length(power_spectrum) / 2) : end));
freq_axis = linspace(0.0, 1 / (2 * dt), length(power_spectrum));

figure, subplot(2, 1, 1);
plot(0 : dt : dt * (length(correlation) - 1), correlation);
title(['Downsampled unresolved velocity correlation DNS100 16 Modes (\tau =', num2str(tau), ')']);
xlabel('Time [s]'), grid minor;
subplot(2, 1, 2);
plot(freq_axis, power_spectrum), title('Power Spectrum')
xlabel('Frequency [Hz]'), grid minor;
print('C:\Users\agustinmartin.picard\Documents\Correlation Functions\DNS100_16modes_downsampled_LMS.png', '-dpng', '-r500');

% DNS300
data = load('C_DNS300_2Modes.mat');
dt = 0.25;
cov_v = data.c;
bt = data.bt;
clear data;

% Estimate the downsampling rate
tau = correlationTimeLMS(cov_v, bt, dt);
tau = max(floor(tau), 1);

% Downsample by tau
% cov_v = cov_v(1 : tau : end, 1 : tau : end);
% bt = bt(1 : tau : end, :);

% Estimate the correlation function and its spectrum
cov_s = smallScaleVelocityCov(cov_v, bt);
cov_s = cov_s(1 : tau : end, 1 : tau : end);
correlation = estimateAutocorrelation(cov_s);
power_spectrum = fftshift(fft([correlation', zeros(size(correlation'))]));
power_spectrum = abs(power_spectrum(floor(length(power_spectrum) / 2) : end));
freq_axis = linspace(0.0, 1 / (2 * dt), length(power_spectrum));

figure, subplot(2, 1, 1);
plot(0 : dt : dt * (length(correlation) - 1), correlation);
title(['Downsampled unresolved velocity correlation DNS300 2 Modes (\tau =', num2str(tau), ')']);
xlabel('Time [s]'), grid minor;
subplot(2, 1, 2);
plot(freq_axis, power_spectrum), title('Power Spectrum')
xlabel('Frequency [Hz]'), grid minor;
print('C:\Users\agustinmartin.picard\Documents\Correlation Functions\DNS300_2modes_downsampled_LMS.png', '-dpng', '-r500');


data = load('C_DNS300_4Modes.mat');
dt = 0.25;
cov_v = data.c;
bt = data.bt;
clear data;

% Estimate the downsampling rate
tau = correlationTimeLMS(cov_v, bt, dt);
tau = max(floor(tau), 1);

% Downsample by tau
% cov_v = cov_v(1 : tau : end, 1 : tau : end);
% bt = bt(1 : tau : end, :);

% Estimate the correlation function and its spectrum
cov_s = smallScaleVelocityCov(cov_v, bt);
cov_s = cov_s(1 : tau : end, 1 : tau : end);
correlation = estimateAutocorrelation(cov_s);
power_spectrum = fftshift(fft([correlation', zeros(size(correlation'))]));
power_spectrum = abs(power_spectrum(floor(length(power_spectrum) / 2) : end));
freq_axis = linspace(0.0, 1 / (2 * dt), length(power_spectrum));

figure, subplot(2, 1, 1);
plot(0 : dt : dt * (length(correlation) - 1), correlation);
title(['Downsampled unresolved velocity correlation DNS300 4 Modes (\tau =', num2str(tau), ')']);
xlabel('Time [s]'), grid minor;
subplot(2, 1, 2);
plot(freq_axis, power_spectrum), title('Power Spectrum')
xlabel('Frequency [Hz]'), grid minor;
print('C:\Users\agustinmartin.picard\Documents\Correlation Functions\DNS300_4modes_downsampled_LMS.png', '-dpng', '-r500');


data = load('C_DNS300_8Modes.mat');
dt = 0.25;
cov_v = data.c;
bt = data.bt;
clear data;

% Estimate the downsampling rate
tau = correlationTimeLMS(cov_v, bt, dt);
tau = max(floor(tau), 1);

% Downsample by tau
% cov_v = cov_v(1 : tau : end, 1 : tau : end);
% bt = bt(1 : tau : end, :);

% Estimate the correlation function and its spectrum
cov_s = smallScaleVelocityCov(cov_v, bt);
cov_s = cov_s(1 : tau : end, 1 : tau : end);
correlation = estimateAutocorrelation(cov_s);
power_spectrum = fftshift(fft([correlation', zeros(size(correlation'))]));
power_spectrum = abs(power_spectrum(floor(length(power_spectrum) / 2) : end));
freq_axis = linspace(0.0, 1 / (2 * dt), length(power_spectrum));

figure, subplot(2, 1, 1);
plot(0 : dt : dt * (length(correlation) - 1), correlation);
title(['Downsampled unresolved velocity correlation DNS300 8 Modes (\tau =', num2str(tau), ')']);
xlabel('Time [s]'), grid minor;
subplot(2, 1, 2);
plot(freq_axis, power_spectrum), title('Power Spectrum')
xlabel('Frequency [Hz]'), grid minor;
print('C:\Users\agustinmartin.picard\Documents\Correlation Functions\DNS300_8modes_downsampled_LMS.png', '-dpng', '-r500');


data = load('C_DNS300_16Modes.mat');
dt = 0.25;
cov_v = data.c;
bt = data.bt;
clear data;

% Estimate the downsampling rate
tau = correlationTimeLMS(cov_v, bt, dt);
tau = max(floor(tau), 1);

% Downsample by tau
% cov_v = cov_v(1 : tau : end, 1 : tau : end);
% bt = bt(1 : tau : end, :);

% Estimate the correlation function and its spectrum
cov_s = smallScaleVelocityCov(cov_v, bt);
cov_s = cov_s(1 : tau : end, 1 : tau : end);
correlation = estimateAutocorrelation(cov_s);
power_spectrum = fftshift(fft([correlation', zeros(size(correlation'))]));
power_spectrum = abs(power_spectrum(floor(length(power_spectrum) / 2) : end));
freq_axis = linspace(0.0, 1 / (2 * dt), length(power_spectrum));

figure, subplot(2, 1, 1);
plot(0 : dt : dt * (length(correlation) - 1), correlation);
title(['Downsampled unresolved velocity correlation DNS300 16 Modes (\tau =', num2str(tau), ')']);
xlabel('Time [s]'), grid minor;
subplot(2, 1, 2);
plot(freq_axis, power_spectrum), title('Power Spectrum')
xlabel('Frequency [Hz]'), grid minor;
print('C:\Users\agustinmartin.picard\Documents\Correlation Functions\DNS300_16modes_downsampled_LMS.png', '-dpng', '-r500');


%% Downsampled version: heterogeneous estimated rate
% DNS100
clear all;
data = load('C_DNS100_2Modes.mat');
dt = 0.05;
cov_v = data.c;
cov_v = cov_v(1 : 10000, 1 : 10000);
bt = data.bt;
clear data;

% Estimate the downsampling rate
tau = simpleCorrelationTime(cov_v, bt, dt);
tau = max(floor(tau), 1);

% Downsample by tau
% cov_v = cov_v(1 : tau : end, 1 : tau : end);
% bt = bt(1 : tau : end, :);

% Estimate the correlation function and its spectrum
cov_s = smallScaleVelocityCov(cov_v, bt);
cov_s = cov_s(1 : tau : end, 1 : tau : end);
correlation = estimateAutocorrelation(cov_s);
power_spectrum = fftshift(fft([correlation', zeros(size(correlation'))]));
power_spectrum = abs(power_spectrum(floor(length(power_spectrum) / 2) : end));
freq_axis = linspace(0.0, 1 / (2 * dt), length(power_spectrum));

figure, subplot(2, 1, 1);
plot(0 : dt : dt * (length(correlation) - 1), correlation);
title(['Downsampled unresolved velocity correlation DNS100 2 Modes (\tau =', num2str(tau), ')']);
xlabel('Time [s]'), grid minor;
subplot(2, 1, 2);
plot(freq_axis, power_spectrum), title('Power Spectrum')
xlabel('Frequency [Hz]'), grid minor;
print('C:\Users\agustinmartin.picard\Documents\Correlation Functions\DNS100_2modes_downsampled_mean.png', '-dpng', '-r500');


data = load('C_DNS100_4Modes.mat');
dt = 0.05;
cov_v = data.c;
cov_v = cov_v(1 : 10000, 1 : 10000);
bt = data.bt;
clear data;

% Estimate the downsampling rate
tau = simpleCorrelationTime(cov_v, bt, dt);
tau = max(floor(tau), 1);

% Downsample by tau
% cov_v = cov_v(1 : tau : end, 1 : tau : end);
% bt = bt(1 : tau : end, :);

% Estimate the correlation function and its spectrum
cov_s = smallScaleVelocityCov(cov_v, bt);
cov_s = cov_s(1 : tau : end, 1 : tau : end);
correlation = estimateAutocorrelation(cov_s);
power_spectrum = fftshift(fft([correlation', zeros(size(correlation'))]));
power_spectrum = abs(power_spectrum(floor(length(power_spectrum) / 2) : end));
freq_axis = linspace(0.0, 1 / (2 * dt), length(power_spectrum));

figure, subplot(2, 1, 1);
plot(0 : dt : dt * (length(correlation) - 1), correlation);
title(['Downsampled unresolved velocity correlation DNS100 4 Modes (\tau =', num2str(tau), ')']);
xlabel('Time [s]'), grid minor;
subplot(2, 1, 2);
plot(freq_axis, power_spectrum), title('Power Spectrum')
xlabel('Frequency [Hz]'), grid minor;
print('C:\Users\agustinmartin.picard\Documents\Correlation Functions\DNS100_4modes_downsampled_mean.png', '-dpng', '-r500');


data = load('C_DNS100_8Modes.mat');
dt = 0.05;
cov_v = data.c;
cov_v = cov_v(1 : 10000, 1 : 10000);
bt = data.bt;
clear data;

% Estimate the downsampling rate
tau = simpleCorrelationTime(cov_v, bt, dt);
tau = max(floor(tau), 1);

% Downsample by tau
% cov_v = cov_v(1 : tau : end, 1 : tau : end);
% bt = bt(1 : tau : end, :);

% Estimate the correlation function and its spectrum
cov_s = smallScaleVelocityCov(cov_v, bt);
cov_s = cov_s(1 : tau : end, 1 : tau : end);
correlation = estimateAutocorrelation(cov_s);
power_spectrum = fftshift(fft([correlation', zeros(size(correlation'))]));
power_spectrum = abs(power_spectrum(floor(length(power_spectrum) / 2) : end));
freq_axis = linspace(0.0, 1 / (2 * dt), length(power_spectrum));

figure, subplot(2, 1, 1);
plot(0 : dt : dt * (length(correlation) - 1), correlation);
title(['Downsampled unresolved velocity correlation DNS100 8 Modes (\tau =', num2str(tau), ')']);
xlabel('Time [s]'), grid minor;
subplot(2, 1, 2);
plot(freq_axis, power_spectrum), title('Power Spectrum')
xlabel('Frequency [Hz]'), grid minor;
print('C:\Users\agustinmartin.picard\Documents\Correlation Functions\DNS100_8modes_downsampled_mean.png', '-dpng', '-r500');


data = load('C_DNS100_16Modes.mat');
dt = 0.05;
cov_v = data.c;
cov_v = cov_v(1 : 10000, 1 : 10000);
bt = data.bt;
clear data;

% Estimate the downsampling rate
tau = simpleCorrelationTime(cov_v, bt, dt);
tau = max(floor(tau), 1);

% Downsample by tau
% cov_v = cov_v(1 : tau : end, 1 : tau : end);
% bt = bt(1 : tau : end, :);

% Estimate the correlation function and its spectrum
cov_s = smallScaleVelocityCov(cov_v, bt);
cov_s = cov_s(1 : tau : end, 1 : tau : end);
correlation = estimateAutocorrelation(cov_s);
power_spectrum = fftshift(fft([correlation', zeros(size(correlation'))]));
power_spectrum = abs(power_spectrum(floor(length(power_spectrum) / 2) : end));
freq_axis = linspace(0.0, 1 / (2 * dt), length(power_spectrum));

figure, subplot(2, 1, 1);
plot(0 : dt : dt * (length(correlation) - 1), correlation);
title(['Downsampled unresolved velocity correlation DNS100 16 Modes (\tau =', num2str(tau), ')']);
xlabel('Time [s]'), grid minor;
subplot(2, 1, 2);
plot(freq_axis, power_spectrum), title('Power Spectrum')
xlabel('Frequency [Hz]'), grid minor;
print('C:\Users\agustinmartin.picard\Documents\Correlation Functions\DNS100_16modes_downsampled_mean.png', '-dpng', '-r500');

% DNS300
data = load('C_DNS300_2Modes.mat');
dt = 0.25;
cov_v = data.c;
bt = data.bt;
clear data;

% Estimate the downsampling rate
tau = simpleCorrelationTime(cov_v, bt, dt);
tau = max(floor(tau), 1);

% Downsample by tau
% cov_v = cov_v(1 : tau : end, 1 : tau : end);
% bt = bt(1 : tau : end, :);

% Estimate the correlation function and its spectrum
cov_s = smallScaleVelocityCov(cov_v, bt);
cov_s = cov_s(1 : tau : end, 1 : tau : end);
correlation = estimateAutocorrelation(cov_s);
power_spectrum = fftshift(fft([correlation', zeros(size(correlation'))]));
power_spectrum = abs(power_spectrum(floor(length(power_spectrum) / 2) : end));
freq_axis = linspace(0.0, 1 / (2 * dt), length(power_spectrum));

figure, subplot(2, 1, 1);
plot(0 : dt : dt * (length(correlation) - 1), correlation);
title(['Downsampled unresolved velocity correlation DNS300 2 Modes (\tau =', num2str(tau), ')']);
xlabel('Time [s]'), grid minor;
subplot(2, 1, 2);
plot(freq_axis, power_spectrum), title('Power Spectrum')
xlabel('Frequency [Hz]'), grid minor;
print('C:\Users\agustinmartin.picard\Documents\Correlation Functions\DNS300_2modes_downsampled_mean.png', '-dpng', '-r500');


data = load('C_DNS300_4Modes.mat');
dt = 0.25;
cov_v = data.c;
bt = data.bt;
clear data;

% Estimate the downsampling rate
tau = simpleCorrelationTime(cov_v, bt, dt);
tau = max(floor(tau), 1);

% Downsample by tau
% cov_v = cov_v(1 : tau : end, 1 : tau : end);
% bt = bt(1 : tau : end, :);

% Estimate the correlation function and its spectrum
cov_s = smallScaleVelocityCov(cov_v, bt);
cov_s = cov_s(1 : tau : end, 1 : tau : end);
correlation = estimateAutocorrelation(cov_s);
power_spectrum = fftshift(fft([correlation', zeros(size(correlation'))]));
power_spectrum = abs(power_spectrum(floor(length(power_spectrum) / 2) : end));
freq_axis = linspace(0.0, 1 / (2 * dt), length(power_spectrum));

figure, subplot(2, 1, 1);
plot(0 : dt : dt * (length(correlation) - 1), correlation);
title(['Downsampled unresolved velocity correlation DNS300 4 Modes (\tau =', num2str(tau), ')']);
xlabel('Time [s]'), grid minor;
subplot(2, 1, 2);
plot(freq_axis, power_spectrum), title('Power Spectrum')
xlabel('Frequency [Hz]'), grid minor;
print('C:\Users\agustinmartin.picard\Documents\Correlation Functions\DNS300_4modes_downsampled_mean.png', '-dpng', '-r500');


data = load('C_DNS300_8Modes.mat');
dt = 0.25;
cov_v = data.c;
bt = data.bt;
clear data;

% Estimate the downsampling rate
tau = simpleCorrelationTime(cov_v, bt, dt);
tau = max(floor(tau), 1);

% Downsample by tau
% cov_v = cov_v(1 : tau : end, 1 : tau : end);
% bt = bt(1 : tau : end, :);

% Estimate the correlation function and its spectrum
cov_s = smallScaleVelocityCov(cov_v, bt);
cov_s = cov_s(1 : tau : end, 1 : tau : end);
correlation = estimateAutocorrelation(cov_s);
power_spectrum = fftshift(fft([correlation', zeros(size(correlation'))]));
power_spectrum = abs(power_spectrum(floor(length(power_spectrum) / 2) : end));
freq_axis = linspace(0.0, 1 / (2 * dt), length(power_spectrum));

figure, subplot(2, 1, 1);
plot(0 : dt : dt * (length(correlation) - 1), correlation);
title(['Downsampled unresolved velocity correlation DNS300 8 Modes (\tau =', num2str(tau), ')']);
xlabel('Time [s]'), grid minor;
subplot(2, 1, 2);
plot(freq_axis, power_spectrum), title('Power Spectrum')
xlabel('Frequency [Hz]'), grid minor;
print('C:\Users\agustinmartin.picard\Documents\Correlation Functions\DNS300_8modes_downsampled_mean.png', '-dpng', '-r500');


data = load('C_DNS300_16Modes.mat');
dt = 0.25;
cov_v = data.c;
bt = data.bt;
clear data;

% Estimate the downsampling rate
tau = simpleCorrelationTime(cov_v, bt, dt);
tau = max(floor(tau), 1);

% Downsample by tau
% cov_v = cov_v(1 : tau : end, 1 : tau : end);
% bt = bt(1 : tau : end, :);

% Estimate the correlation function and its spectrum
cov_s = smallScaleVelocityCov(cov_v, bt);
cov_s = cov_s(1 : tau : end, 1 : tau : end);
correlation = estimateAutocorrelation(cov_s);
power_spectrum = fftshift(fft([correlation', zeros(size(correlation'))]));
power_spectrum = abs(power_spectrum(floor(length(power_spectrum) / 2) : end));
freq_axis = linspace(0.0, 1 / (2 * dt), length(power_spectrum));

figure, subplot(2, 1, 1);
plot(0 : dt : dt * (length(correlation) - 1), correlation);
title(['Downsampled unresolved velocity correlation DNS300 16 Modes (\tau =', num2str(tau), ')']);
xlabel('Time [s]'), grid minor;
subplot(2, 1, 2);
plot(freq_axis, power_spectrum), title('Power Spectrum')
xlabel('Frequency [Hz]'), grid minor;
print('C:\Users\agustinmartin.picard\Documents\Correlation Functions\DNS300_16modes_downsampled_mean.png', '-dpng', '-r500');


%% Downsampled version: cut estimated rate
% DNS100
clear all;
data = load('C_DNS100_2Modes.mat');
dt = 0.05;
cov_v = data.c;
cov_v = cov_v(1 : 10000, 1 : 10000);
bt = data.bt;
clear data;

% Estimate the downsampling rate
tau = correlationTimeCut(cov_v, bt);
tau = max(floor(tau), 1);

% Downsample by tau
% cov_v = cov_v(1 : tau : end, 1 : tau : end);
% bt = bt(1 : tau : end, :);

% Estimate the correlation function and its spectrum
cov_s = smallScaleVelocityCov(cov_v, bt);
cov_s = cov_s(1 : tau : end, 1 : tau : end);
correlation = estimateAutocorrelation(cov_s);
power_spectrum = fftshift(fft([correlation', zeros(size(correlation'))]));
power_spectrum = abs(power_spectrum(floor(length(power_spectrum) / 2) : end));
freq_axis = linspace(0.0, 1 / (2 * dt), length(power_spectrum));

figure, subplot(2, 1, 1);
plot(0 : dt : dt * (length(correlation) - 1), correlation);
title(['Downsampled unresolved velocity correlation DNS100 2 Modes (\tau =', num2str(tau), ')']);
xlabel('Time [s]'), grid minor;
subplot(2, 1, 2);
plot(freq_axis, power_spectrum), title('Power Spectrum')
xlabel('Frequency [Hz]'), grid minor;
print('C:\Users\agustinmartin.picard\Documents\Correlation Functions\DNS100_2modes_downsampled_cut.png', '-dpng', '-r500');


clear all;
data = load('C_DNS100_4Modes.mat');
dt = 0.05;
cov_v = data.c;
cov_v = cov_v(1 : 10000, 1 : 10000);
bt = data.bt;
clear data;

% Estimate the downsampling rate
tau = correlationTimeCut(cov_v, bt);
tau = max(floor(tau), 1);

% Downsample by tau
% cov_v = cov_v(1 : tau : end, 1 : tau : end);
% bt = bt(1 : tau : end, :);

% Estimate the correlation function and its spectrum
cov_s = smallScaleVelocityCov(cov_v, bt);
cov_s = cov_s(1 : tau : end, 1 : tau : end);
correlation = estimateAutocorrelation(cov_s);
power_spectrum = fftshift(fft([correlation', zeros(size(correlation'))]));
power_spectrum = abs(power_spectrum(floor(length(power_spectrum) / 2) : end));
freq_axis = linspace(0.0, 1 / (2 * dt), length(power_spectrum));

figure, subplot(2, 1, 1);
plot(0 : dt : dt * (length(correlation) - 1), correlation);
title(['Downsampled unresolved velocity correlation DNS100 4 Modes (\tau =', num2str(tau), ')']);
xlabel('Time [s]'), grid minor;
subplot(2, 1, 2);
plot(freq_axis, power_spectrum), title('Power Spectrum')
xlabel('Frequency [Hz]'), grid minor;
print('C:\Users\agustinmartin.picard\Documents\Correlation Functions\DNS100_4modes_downsampled_cut.png', '-dpng', '-r500');


clear all;
data = load('C_DNS100_8Modes.mat');
dt = 0.05;
cov_v = data.c;
cov_v = cov_v(1 : 10000, 1 : 10000);
bt = data.bt;
clear data;

% Estimate the downsampling rate
tau = correlationTimeCut(cov_v, bt);
tau = max(floor(tau), 1);

% Downsample by tau
% cov_v = cov_v(1 : tau : end, 1 : tau : end);
% bt = bt(1 : tau : end, :);

% Estimate the correlation function and its spectrum
cov_s = smallScaleVelocityCov(cov_v, bt);
cov_s = cov_s(1 : tau : end, 1 : tau : end);
correlation = estimateAutocorrelation(cov_s);
power_spectrum = fftshift(fft([correlation', zeros(size(correlation'))]));
power_spectrum = abs(power_spectrum(floor(length(power_spectrum) / 2) : end));
freq_axis = linspace(0.0, 1 / (2 * dt), length(power_spectrum));

figure, subplot(2, 1, 1);
plot(0 : dt : dt * (length(correlation) - 1), correlation);
title(['Downsampled unresolved velocity correlation DNS100 8 Modes (\tau =', num2str(tau), ')']);
xlabel('Time [s]'), grid minor;
subplot(2, 1, 2);
plot(freq_axis, power_spectrum), title('Power Spectrum')
xlabel('Frequency [Hz]'), grid minor;
print('C:\Users\agustinmartin.picard\Documents\Correlation Functions\DNS100_8modes_downsampled_cut.png', '-dpng', '-r500');


clear all;
data = load('C_DNS100_16Modes.mat');
dt = 0.05;
cov_v = data.c;
cov_v = cov_v(1 : 10000, 1 : 10000);
bt = data.bt;
clear data;

% Estimate the downsampling rate
tau = correlationTimeCut(cov_v, bt);
tau = max(floor(tau), 1);

% Downsample by tau
% cov_v = cov_v(1 : tau : end, 1 : tau : end);
% bt = bt(1 : tau : end, :);

% Estimate the correlation function and its spectrum
cov_s = smallScaleVelocityCov(cov_v, bt);
cov_s = cov_s(1 : tau : end, 1 : tau : end);
correlation = estimateAutocorrelation(cov_s);
power_spectrum = fftshift(fft([correlation', zeros(size(correlation'))]));
power_spectrum = abs(power_spectrum(floor(length(power_spectrum) / 2) : end));
freq_axis = linspace(0.0, 1 / (2 * dt), length(power_spectrum));

figure, subplot(2, 1, 1);
plot(0 : dt : dt * (length(correlation) - 1), correlation);
title(['Downsampled unresolved velocity correlation DNS100 16 Modes (\tau =', num2str(tau), ')']);
xlabel('Time [s]'), grid minor;
subplot(2, 1, 2);
plot(freq_axis, power_spectrum), title('Power Spectrum')
xlabel('Frequency [Hz]'), grid minor;
print('C:\Users\agustinmartin.picard\Documents\Correlation Functions\DNS100_16modes_downsampled_cut.png', '-dpng', '-r500');

% DNS300
data = load('C_DNS300_2Modes.mat');
dt = 0.25;
cov_v = data.c;
bt = data.bt;
clear data;

% Estimate the downsampling rate
tau = correlationTimeCut(cov_v, bt);
tau = max(floor(tau), 1);

% Downsample by tau
% cov_v = cov_v(1 : tau : end, 1 : tau : end);
% bt = bt(1 : tau : end, :);

% Estimate the correlation function and its spectrum
cov_s = smallScaleVelocityCov(cov_v, bt);
cov_s = cov_s(1 : tau : end, 1 : tau : end);
correlation = estimateAutocorrelation(cov_s);
power_spectrum = fftshift(fft([correlation', zeros(size(correlation'))]));
power_spectrum = abs(power_spectrum(floor(length(power_spectrum) / 2) : end));
freq_axis = linspace(0.0, 1 / (2 * dt), length(power_spectrum));

figure, subplot(2, 1, 1);
plot(0 : dt : dt * (length(correlation) - 1), correlation);
title(['Downsampled unresolved velocity correlation DNS300 2 Modes (\tau =', num2str(tau), ')']);
xlabel('Time [s]'), grid minor;
subplot(2, 1, 2);
plot(freq_axis, power_spectrum), title('Power Spectrum')
xlabel('Frequency [Hz]'), grid minor;
print('C:\Users\agustinmartin.picard\Documents\Correlation Functions\DNS300_2modes_downsampled_cut.png', '-dpng', '-r500');


data = load('C_DNS300_4Modes.mat');
dt = 0.25;
cov_v = data.c;
bt = data.bt;
clear data;

% Estimate the downsampling rate
tau = correlationTimeCut(cov_v, bt);
tau = max(floor(tau), 1);

% Downsample by tau
% cov_v = cov_v(1 : tau : end, 1 : tau : end);
% bt = bt(1 : tau : end, :);

% Estimate the correlation function and its spectrum
cov_s = smallScaleVelocityCov(cov_v, bt);
cov_s = cov_s(1 : tau : end, 1 : tau : end);
correlation = estimateAutocorrelation(cov_s);
power_spectrum = fftshift(fft([correlation', zeros(size(correlation'))]));
power_spectrum = abs(power_spectrum(floor(length(power_spectrum) / 2) : end));
freq_axis = linspace(0.0, 1 / (2 * dt), length(power_spectrum));

figure, subplot(2, 1, 1);
plot(0 : dt : dt * (length(correlation) - 1), correlation);
title(['Downsampled unresolved velocity correlation DNS300 4 Modes (\tau =', num2str(tau), ')']);
xlabel('Time [s]'), grid minor;
subplot(2, 1, 2);
plot(freq_axis, power_spectrum), title('Power Spectrum')
xlabel('Frequency [Hz]'), grid minor;
print('C:\Users\agustinmartin.picard\Documents\Correlation Functions\DNS300_4modes_downsampled_cut.png', '-dpng', '-r500');


data = load('C_DNS300_8Modes.mat');
dt = 0.25;
cov_v = data.c;
bt = data.bt;
clear data;

% Estimate the downsampling rate
tau = correlationTimeCut(cov_v, bt);
tau = max(floor(tau), 1);

% Downsample by tau
% cov_v = cov_v(1 : tau : end, 1 : tau : end);
% bt = bt(1 : tau : end, :);

% Estimate the correlation function and its spectrum
cov_s = smallScaleVelocityCov(cov_v, bt);
cov_s = cov_s(1 : tau : end, 1 : tau : end);
correlation = estimateAutocorrelation(cov_s);
power_spectrum = fftshift(fft([correlation', zeros(size(correlation'))]));
power_spectrum = abs(power_spectrum(floor(length(power_spectrum) / 2) : end));
freq_axis = linspace(0.0, 1 / (2 * dt), length(power_spectrum));

figure, subplot(2, 1, 1);
plot(0 : dt : dt * (length(correlation) - 1), correlation);
title(['Downsampled unresolved velocity correlation DNS300 8 Modes (\tau =', num2str(tau), ')']);
xlabel('Time [s]'), grid minor;
subplot(2, 1, 2);
plot(freq_axis, power_spectrum), title('Power Spectrum')
xlabel('Frequency [Hz]'), grid minor;
print('C:\Users\agustinmartin.picard\Documents\Correlation Functions\DNS300_8modes_downsampled_cut.png', '-dpng', '-r500');


data = load('C_DNS300_16Modes.mat');
dt = 0.25;
cov_v = data.c;
bt = data.bt;
clear data;

% Estimate the downsampling rate
tau = correlationTimeCut(cov_v, bt);
tau = max(floor(tau), 1);

% Downsample by tau
% cov_v = cov_v(1 : tau : end, 1 : tau : end);
% bt = bt(1 : tau : end, :);

% Estimate the correlation function and its spectrum
cov_s = smallScaleVelocityCov(cov_v, bt);
cov_s = cov_s(1 : tau : end, 1 : tau : end);
correlation = estimateAutocorrelation(cov_s);
power_spectrum = fftshift(fft([correlation', zeros(size(correlation'))]));
power_spectrum = abs(power_spectrum(floor(length(power_spectrum) / 2) : end));
freq_axis = linspace(0.0, 1 / (2 * dt), length(power_spectrum));

figure, subplot(2, 1, 1);
plot(0 : dt : dt * (length(correlation) - 1), correlation);
title(['Downsampled unresolved velocity correlation DNS300 16 Modes (\tau =', num2str(tau), ')']);
xlabel('Time [s]'), grid minor;
subplot(2, 1, 2);
plot(freq_axis, power_spectrum), title('Power Spectrum')
xlabel('Frequency [Hz]'), grid minor;
print('C:\Users\agustinmartin.picard\Documents\Correlation Functions\DNS300_16modes_downsampled_cut.png', '-dpng', '-r500');


%% Downsampled version: shannon estimated rate
% DNS100
clear all;
data = load('C_DNS100_2Modes.mat');
dt = 0.05;
cov_v = data.c;
cov_v = cov_v(1 : 10000, 1 : 10000);
bt = data.bt;
nb_modes = 2;
clear data;

[~,S]=eig(cov_v);
lambda=diag(S);clear S % singular values : energy of each modes
lambda=lambda(end:-1:1);
lambda=max(lambda,0);
lambda=lambda(1 : nb_modes);

param.dt = dt;
param.decor_by_subsampl.test_fct = 'b';
param.decor_by_subsampl.spectrum_threshold = 1e-6;

% Estimate the downsampling rate
tau = fct_cut_frequency(bt, lambda, param);
tau = max(floor(tau), 1);

% Downsample by tau
% cov_v = cov_v(1 : tau : end, 1 : tau : end);
% bt = bt(1 : tau : end, :);

% Estimate the correlation function and its spectrum
cov_s = smallScaleVelocityCov(cov_v, bt);
cov_s = cov_s(1 : tau : end, 1 : tau : end);
correlation = estimateAutocorrelation(cov_s);
power_spectrum = fftshift(fft([correlation', zeros(size(correlation'))]));
power_spectrum = abs(power_spectrum(floor(length(power_spectrum) / 2) : end));
freq_axis = linspace(0.0, 1 / (2 * dt), length(power_spectrum));

figure, subplot(2, 1, 1);
plot(0 : dt : dt * (length(correlation) - 1), correlation);
title(['Downsampled unresolved velocity correlation DNS100 2 Modes (\tau =', num2str(tau), ')']);
xlabel('Time [s]'), grid minor;
subplot(2, 1, 2);
plot(freq_axis, power_spectrum), title('Power Spectrum')
xlabel('Frequency [Hz]'), grid minor;
print('C:\Users\agustinmartin.picard\Documents\Correlation Functions\DNS100_2modes_downsampled_shannon.png', '-dpng', '-r500');


clear all;
data = load('C_DNS100_4Modes.mat');
dt = 0.05;
cov_v = data.c;
cov_v = cov_v(1 : 10000, 1 : 10000);
bt = data.bt;
nb_modes = 4;
clear data;

[~,S]=eig(cov_v);
lambda=diag(S);clear S % singular values : energy of each modes
lambda=lambda(end:-1:1);
lambda=max(lambda,0);
lambda=lambda(1 : nb_modes);

param.dt = dt;
param.decor_by_subsampl.test_fct = 'b';
param.decor_by_subsampl.spectrum_threshold = 1e-6;

% Estimate the downsampling rate
tau = fct_cut_frequency(bt, lambda, param);
tau = max(floor(tau), 1);

% Downsample by tau
% cov_v = cov_v(1 : tau : end, 1 : tau : end);
% bt = bt(1 : tau : end, :);

% Estimate the correlation function and its spectrum
cov_s = smallScaleVelocityCov(cov_v, bt);
cov_s = cov_s(1 : tau : end, 1 : tau : end);
correlation = estimateAutocorrelation(cov_s);
power_spectrum = fftshift(fft([correlation', zeros(size(correlation'))]));
power_spectrum = abs(power_spectrum(floor(length(power_spectrum) / 2) : end));
freq_axis = linspace(0.0, 1 / (2 * dt), length(power_spectrum));

figure, subplot(2, 1, 1);
plot(0 : dt : dt * (length(correlation) - 1), correlation);
title(['Downsampled unresolved velocity correlation DNS100 4 Modes (\tau =', num2str(tau), ')']);
xlabel('Time [s]'), grid minor;
subplot(2, 1, 2);
plot(freq_axis, power_spectrum), title('Power Spectrum')
xlabel('Frequency [Hz]'), grid minor;
print('C:\Users\agustinmartin.picard\Documents\Correlation Functions\DNS100_4modes_downsampled_shannon.png', '-dpng', '-r500');


clear all;
data = load('C_DNS100_8Modes.mat');
dt = 0.05;
cov_v = data.c;
cov_v = cov_v(1 : 10000, 1 : 10000);
bt = data.bt;
nb_modes = 8;
clear data;

[~,S]=eig(cov_v);
lambda=diag(S);clear S % singular values : energy of each modes
lambda=lambda(end:-1:1);
lambda=max(lambda,0);
lambda=lambda(1 : nb_modes);

param.dt = dt;
param.decor_by_subsampl.test_fct = 'b';
param.decor_by_subsampl.spectrum_threshold = 1e-6;

% Estimate the downsampling rate
tau = fct_cut_frequency(bt, lambda, param);
tau = max(floor(tau), 1);

% Downsample by tau
% cov_v = cov_v(1 : tau : end, 1 : tau : end);
% bt = bt(1 : tau : end, :);

% Estimate the correlation function and its spectrum
cov_s = smallScaleVelocityCov(cov_v, bt);
cov_s = cov_s(1 : tau : end, 1 : tau : end);
correlation = estimateAutocorrelation(cov_s);
power_spectrum = fftshift(fft([correlation', zeros(size(correlation'))]));
power_spectrum = abs(power_spectrum(floor(length(power_spectrum) / 2) : end));
freq_axis = linspace(0.0, 1 / (2 * dt), length(power_spectrum));

figure, subplot(2, 1, 1);
plot(0 : dt : dt * (length(correlation) - 1), correlation);
title(['Downsampled unresolved velocity correlation DNS100 8 Modes (\tau =', num2str(tau), ')']);
xlabel('Time [s]'), grid minor;
subplot(2, 1, 2);
plot(freq_axis, power_spectrum), title('Power Spectrum')
xlabel('Frequency [Hz]'), grid minor;
print('C:\Users\agustinmartin.picard\Documents\Correlation Functions\DNS100_8modes_downsampled_shannon.png', '-dpng', '-r500');


clear all;
data = load('C_DNS100_16Modes.mat');
dt = 0.05;
cov_v = data.c;
cov_v = cov_v(1 : 10000, 1 : 10000);
bt = data.bt;
nb_modes = 16;
clear data;

[~,S]=eig(cov_v);
lambda=diag(S);clear S % singular values : energy of each modes
lambda=lambda(end:-1:1);
lambda=max(lambda,0);
lambda=lambda(1 : nb_modes);

param.dt = dt;
param.decor_by_subsampl.test_fct = 'b';
param.decor_by_subsampl.spectrum_threshold = 1e-6;

% Estimate the downsampling rate
tau = fct_cut_frequency(bt, lambda, param);
tau = max(floor(tau), 1);

% Downsample by tau
% cov_v = cov_v(1 : tau : end, 1 : tau : end);
% bt = bt(1 : tau : end, :);

% Estimate the correlation function and its spectrum
cov_s = smallScaleVelocityCov(cov_v, bt);
cov_s = cov_s(1 : tau : end, 1 : tau : end);
correlation = estimateAutocorrelation(cov_s);
power_spectrum = fftshift(fft([correlation', zeros(size(correlation'))]));
power_spectrum = abs(power_spectrum(floor(length(power_spectrum) / 2) : end));
freq_axis = linspace(0.0, 1 / (2 * dt), length(power_spectrum));

figure, subplot(2, 1, 1);
plot(0 : dt : dt * (length(correlation) - 1), correlation);
title(['Downsampled unresolved velocity correlation DNS100 16 Modes (\tau =', num2str(tau), ')']);
xlabel('Time [s]'), grid minor;
subplot(2, 1, 2);
plot(freq_axis, power_spectrum), title('Power Spectrum')
xlabel('Frequency [Hz]'), grid minor;
print('C:\Users\agustinmartin.picard\Documents\Correlation Functions\DNS100_16modes_downsampled_shannon.png', '-dpng', '-r500');

% DNS300
clear all;
data = load('C_DNS300_2Modes.mat');
dt = 0.25;
cov_v = data.c;
bt = data.bt;
nb_modes = 2;
clear data;

[~,S]=eig(cov_v);
lambda=diag(S);clear S % singular values : energy of each modes
lambda=lambda(end:-1:1);
lambda=max(lambda,0);
lambda=lambda(1 : nb_modes);

param.dt = dt;
param.decor_by_subsampl.test_fct = 'b';
param.decor_by_subsampl.spectrum_threshold = 1e-4;

% Estimate the downsampling rate
tau = fct_cut_frequency(bt, lambda, param);
tau = max(floor(tau), 1);

% Downsample by tau
% cov_v = cov_v(1 : tau : end, 1 : tau : end);
% bt = bt(1 : tau : end, :);

% Estimate the correlation function and its spectrum
cov_s = smallScaleVelocityCov(cov_v, bt);
cov_s = cov_s(1 : tau : end, 1 : tau : end);
correlation = estimateAutocorrelation(cov_s);
power_spectrum = fftshift(fft([correlation', zeros(size(correlation'))]));
power_spectrum = abs(power_spectrum(floor(length(power_spectrum) / 2) : end));
freq_axis = linspace(0.0, 1 / (2 * dt), length(power_spectrum));

figure, subplot(2, 1, 1);
plot(0 : dt : dt * (length(correlation) - 1), correlation);
title(['Downsampled unresolved velocity correlation DNS300 2 Modes (\tau =', num2str(tau), ')']);
xlabel('Time [s]'), grid minor;
subplot(2, 1, 2);
plot(freq_axis, power_spectrum), title('Power Spectrum')
xlabel('Frequency [Hz]'), grid minor;
print('C:\Users\agustinmartin.picard\Documents\Correlation Functions\DNS300_2modes_downsampled_shannon.png', '-dpng', '-r500');


clear all;
data = load('C_DNS300_4Modes.mat');
dt = 0.25;
cov_v = data.c;
bt = data.bt;
nb_modes = 4;
clear data;

[~,S]=eig(cov_v);
lambda=diag(S);clear S % singular values : energy of each modes
lambda=lambda(end:-1:1);
lambda=max(lambda,0);
lambda=lambda(1 : nb_modes);

param.dt = dt;
param.decor_by_subsampl.test_fct = 'b';
param.decor_by_subsampl.spectrum_threshold = 1e-4;

% Estimate the downsampling rate
tau = fct_cut_frequency(bt, lambda, param);
tau = max(floor(tau), 1);

% Downsample by tau
% cov_v = cov_v(1 : tau : end, 1 : tau : end);
% bt = bt(1 : tau : end, :);

% Estimate the correlation function and its spectrum
cov_s = smallScaleVelocityCov(cov_v, bt);
cov_s = cov_s(1 : tau : end, 1 : tau : end);
correlation = estimateAutocorrelation(cov_s);
power_spectrum = fftshift(fft([correlation', zeros(size(correlation'))]));
power_spectrum = abs(power_spectrum(floor(length(power_spectrum) / 2) : end));
freq_axis = linspace(0.0, 1 / (2 * dt), length(power_spectrum));

figure, subplot(2, 1, 1);
plot(0 : dt : dt * (length(correlation) - 1), correlation);
title(['Downsampled unresolved velocity correlation DNS300 4 Modes (\tau =', num2str(tau), ')']);
xlabel('Time [s]'), grid minor;
subplot(2, 1, 2);
plot(freq_axis, power_spectrum), title('Power Spectrum')
xlabel('Frequency [Hz]'), grid minor;
print('C:\Users\agustinmartin.picard\Documents\Correlation Functions\DNS300_4modes_downsampled_shannon.png', '-dpng', '-r500');


clear all;
data = load('C_DNS300_8Modes.mat');
dt = 0.25;
cov_v = data.c;
bt = data.bt;
nb_modes = 8;
clear data;

[~,S]=eig(cov_v);
lambda=diag(S);clear S % singular values : energy of each modes
lambda=lambda(end:-1:1);
lambda=max(lambda,0);
lambda=lambda(1 : nb_modes);

param.dt = dt;
param.decor_by_subsampl.test_fct = 'b';
param.decor_by_subsampl.spectrum_threshold = 1e-4;

% Estimate the downsampling rate
tau = fct_cut_frequency(bt, lambda, param);
tau = max(floor(tau), 1);

% Downsample by tau
% cov_v = cov_v(1 : tau : end, 1 : tau : end);
% bt = bt(1 : tau : end, :);

% Estimate the correlation function and its spectrum
cov_s = smallScaleVelocityCov(cov_v, bt);
cov_s = cov_s(1 : tau : end, 1 : tau : end);
correlation = estimateAutocorrelation(cov_s);
power_spectrum = fftshift(fft([correlation', zeros(size(correlation'))]));
power_spectrum = abs(power_spectrum(floor(length(power_spectrum) / 2) : end));
freq_axis = linspace(0.0, 1 / (2 * dt), length(power_spectrum));

figure, subplot(2, 1, 1);
plot(0 : dt : dt * (length(correlation) - 1), correlation);
title(['Downsampled unresolved velocity correlation DNS300 8 Modes (\tau =', num2str(tau), ')']);
xlabel('Time [s]'), grid minor;
subplot(2, 1, 2);
plot(freq_axis, power_spectrum), title('Power Spectrum')
xlabel('Frequency [Hz]'), grid minor;
print('C:\Users\agustinmartin.picard\Documents\Correlation Functions\DNS300_8modes_downsampled_shannon.png', '-dpng', '-r500');


clear all;
data = load('C_DNS300_16Modes.mat');
dt = 0.25;
cov_v = data.c;
bt = data.bt;
nb_modes = 16;
clear data;

[~,S]=eig(cov_v);
lambda=diag(S);clear S % singular values : energy of each modes
lambda=lambda(end:-1:1);
lambda=max(lambda,0);
lambda=lambda(1 : nb_modes);

param.dt = dt;
param.decor_by_subsampl.test_fct = 'b';
param.decor_by_subsampl.spectrum_threshold = 1e-4;

% Estimate the downsampling rate
tau = fct_cut_frequency(bt, lambda, param);
tau = max(floor(tau), 1);

% Downsample by tau
% cov_v = cov_v(1 : tau : end, 1 : tau : end);
% bt = bt(1 : tau : end, :);

% Estimate the correlation function and its spectrum
cov_s = smallScaleVelocityCov(cov_v, bt);
cov_s = cov_s(1 : tau : end, 1 : tau : end);
correlation = estimateAutocorrelation(cov_s);
power_spectrum = fftshift(fft([correlation', zeros(size(correlation'))]));
power_spectrum = abs(power_spectrum(floor(length(power_spectrum) / 2) : end));
freq_axis = linspace(0.0, 1 / (2 * dt), length(power_spectrum));

figure, subplot(2, 1, 1);
plot(0 : dt : dt * (length(correlation) - 1), correlation);
title(['Downsampled unresolved velocity correlation DNS300 16 Modes (\tau =', num2str(tau), ')']);
xlabel('Time [s]'), grid minor;
subplot(2, 1, 2);
plot(freq_axis, power_spectrum), title('Power Spectrum')
xlabel('Frequency [Hz]'), grid minor;
print('C:\Users\agustinmartin.picard\Documents\Correlation Functions\DNS300_16modes_downsampled_shannon.png', '-dpng', '-r500');
