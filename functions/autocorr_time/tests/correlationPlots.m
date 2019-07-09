%% 
clear all, close all, clc;

%% DNS100
data = load('C_DNS100_2Modes.mat');
dt = 0.05;
cov_v = data.c;
bt = data.bt;
clear data;
cov_s = smallScaleVelocityCov(cov_v, bt);
correlation = estimateAutocorrelation(cov_s);
correlation = correlation(1 : 10000);
power_spectrum = fftshift(fft([correlation', zeros(size(correlation'))]));
power_spectrum = abs(power_spectrum(floor(length(power_spectrum) / 2) : end));
freq_axis = linspace(0.0, 1 / (2 * dt), length(power_spectrum));

figure, subplot(2, 1, 1);
plot(0 : dt : dt * (length(correlation) - 1), correlation), title('Autocorrelation DNS100 2 Modes');
xlabel('Time [s]'), grid minor;
subplot(2, 1, 2);
plot(freq_axis, power_spectrum), title('Power Spectrum DNS100 2 Modes')
xlabel('Frequency [Hz]'), grid minor;

data = load('C_DNS100_4Modes.mat');
dt = 0.05;
cov_v = data.c;
bt = data.bt;
clear data;
cov_s = smallScaleVelocityCov(cov_v, bt);
correlation = estimateAutocorrelation(cov_s);
correlation = correlation(1 : 10000);
power_spectrum = fftshift(fft([correlation', zeros(size(correlation'))]));
power_spectrum = abs(power_spectrum(floor(length(power_spectrum) / 2) : end));

figure, subplot(2, 1, 1);
plot(0 : dt : dt * (length(correlation) - 1), correlation), title('Autocorrelation DNS100 4 Modes');
xlabel('Time [s]'), grid minor;
subplot(2, 1, 2);
plot(freq_axis, power_spectrum), title('Power Spectrum DNS100 4 Modes')
xlabel('Frequency [Hz]'), grid minor;

data = load('C_DNS100_8Modes.mat');
dt = 0.05;
cov_v = data.c;
bt = data.bt;
clear data;
cov_s = smallScaleVelocityCov(cov_v, bt);
correlation = estimateAutocorrelation(cov_s);
correlation = correlation(1 : 10000);
power_spectrum = fftshift(fft([correlation', zeros(size(correlation'))]));
power_spectrum = abs(power_spectrum(floor(length(power_spectrum) / 2) : end));

figure, subplot(2, 1, 1);
plot(0 : dt : dt * (length(correlation) - 1), correlation), title('Autocorrelation DNS100 8 Modes');
xlabel('Time [s]'), grid minor;
subplot(2, 1, 2);
plot(freq_axis, power_spectrum), title('Power Spectrum DNS100 8 Modes')
xlabel('Frequency [Hz]'), grid minor;

data = load('C_DNS100_16Modes.mat');
dt = 0.05;
cov_v = data.c;
bt = data.bt;
clear data;
cov_s = smallScaleVelocityCov(cov_v, bt);
correlation = estimateAutocorrelation(cov_s);
correlation = correlation(1 : 10000);
power_spectrum = fftshift(fft([correlation', zeros(size(correlation'))]));
power_spectrum = abs(power_spectrum(floor(length(power_spectrum) / 2) : end));

figure, subplot(2, 1, 1);
plot(0 : dt : dt * (length(correlation) - 1), correlation), title('Autocorrelation DNS100 16 Modes');
xlabel('Time [s]'), grid minor;
subplot(2, 1, 2);
plot(freq_axis, power_spectrum), title('Power Spectrum DNS100 16 Modes')
xlabel('Frequency [Hz]'), grid minor;

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
plot(0 : dt : dt * (length(correlation) - 1), correlation), title('Autocorrelation DNS300 2 Modes');
xlabel('Time [s]'), grid minor;
subplot(2, 1, 2);
plot(freq_axis, power_spectrum), title('Power Spectrum DNS300 2 Modes')
xlabel('Frequency [Hz]'), grid minor;

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
plot(0 : dt : dt * (length(correlation) - 1), correlation), title('Autocorrelation DNS300 4 Modes');
xlabel('Time [s]'), grid minor;
subplot(2, 1, 2);
plot(freq_axis, power_spectrum), title('Power Spectrum DNS300 4 Modes')
xlabel('Frequency [Hz]'), grid minor;

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
plot(0 : dt : dt * (length(correlation) - 1), correlation), title('Autocorrelation DNS300 8 Modes');
xlabel('Time [s]'), grid minor;
subplot(2, 1, 2);
plot(freq_axis, power_spectrum), title('Power Spectrum DNS300 8 Modes')
xlabel('Frequency [Hz]'), grid minor;

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
plot(0 : dt : dt * (length(correlation) - 1), correlation), title('Autocorrelation DNS300 16 Modes');
xlabel('Time [s]'), grid minor;
subplot(2, 1, 2);
plot(freq_axis, power_spectrum), title('Power Spectrum DNS300 16 Modes')
xlabel('Frequency [Hz]'), grid minor;
