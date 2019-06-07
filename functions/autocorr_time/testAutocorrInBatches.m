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

mode = 'global';
autocorrTime = autocorrelationTimeInBatches(cov_v, bt, mode);

% Plot the autocorrelation estimation as a function of the period
figure, plot(autocorrTime), grid minor;
title('Autocorrelation time (DNS100 - 2 modes - global \sigma)');
xlabel('Period index'), ylabel('\tau_{corr} [s amples / s]');

% Estimate the different means
meanACTime = mean(autocorrTime)
weightedACTime = weightedACTimeMean(autocorrTime)

%% Comparison of the spectrum of the autocorr and the cov matrix
clear all, close all, clc;

data = load('C_DNS100_2Modes.mat'); 
dt = 0.05;
% data = load('C_DNS300_2Modes.mat');
% dt = 0.25;
nyquistFreq = 1 / (2 * dt);
cov_v = data.c;
bt = data.bt;
clear data;

% Calculate the unresolved velocity field
N = size(cov_v, 2);
cov_s = smallScaleVelocityCov(cov_v, bt);
var_v = trace(cov_v);
var_s = trace(cov_s);

% Estimate the autocorrelation function
autocorrFunction = fullAutocorrelation(cov_s);
% autocorrFunction = estimateAutocorrelation(cov_s);

% Calculate the fft and estimate the period as the biggest mode
% powerSpectrumAC = fftshift(fft(autocorrFunction));
powerSpectrumAC = fftshift(fft([autocorrFunction', zeros(size(autocorrFunction'))]));
powerSpectrumAC = abs(powerSpectrumAC(floor(length(powerSpectrumAC) / 2) : end)); % just keep half of the amplitude
powerSpectrumAC = powerSpectrumAC ./ N;

[~, frequenceCentral] = max(powerSpectrumAC);
periodAC = ceil(N / frequenceCentral);

% Now for the spectrum of the covariance matrix
% Get the antidiagonal
antiDiag = diag(fliplr(cov_s));
% antiDiag = antiDiag(length(antiDiag) / 2 : end);

% Calculate its spectrum
% powerSpectrumCov = fftshift(fft(antiDiag));
powerSpectrumCov = fftshift(fft([antiDiag', zeros(size(antiDiag'))]));
powerSpectrumCov = abs(powerSpectrumCov(floor(length(powerSpectrumCov) / 2) : end)); % just keep half of the amplitude
powerSpectrumCov = powerSpectrumCov ./ var_s;

% Get the central frequency
[~, frequence_central] = max(powerSpectrumCov);
periodCov = ceil(N / frequence_central);

% Finally, for the spectrum of the full matrix
antiDiagFull = diag(fliplr(cov_v));
% antiDiagFull = antiDiagFull(length(antiDiagFull) / 2 : end);

% Calculate its spectrum
% powerSpectrumFull = fftshift(fft(antiDiagFull));
powerSpectrumFull = fftshift(fft([antiDiagFull', zeros(size(antiDiagFull'))]));
powerSpectrumFull = abs(powerSpectrumFull(floor(length(powerSpectrumFull) / 2) : end)); % just keep half of the amplitude
powerSpectrumFull = powerSpectrumFull ./ var_v;

% Get the central frequency
[~, frequenceFull] = max(powerSpectrumFull);
periodFull = ceil(N / frequenceFull);

% Estimate the autocorrelation function
autocorrFunctionFull = fullAutocorrelation(cov_v);
% autocorrFunctionFull = estimateAutocorrelation(cov_v);

% Calculate the fft and estimate the period as the biggest mode
% powerSpectrumACFull = fftshift(fft(autocorrFunctionFull));
powerSpectrumACFull = fftshift(fft([autocorrFunctionFull', zeros(size(autocorrFunctionFull'))]));
powerSpectrumACFull = abs(powerSpectrumACFull(floor(length(powerSpectrumACFull) / 2) : end)); % just keep half of the amplitude
powerSpectrumACFull = powerSpectrumACFull ./ N;

[~, frequenceCentralFull] = max(powerSpectrumACFull);
periodACFull = ceil(N / frequenceCentralFull);

% Plot both in subplots
figure, subplot(4, 1, 1);
plot(linspace(0.0, nyquistFreq, length(powerSpectrumCov)), powerSpectrumCov);
xlabel('Normalized frequency'), ylabel('Power [m^2 / s^2]');
title('Spectrum of covariance matrix (DNS300 - 4 modes)');
grid minor;
subplot(4, 1, 2);
plot(linspace(0.0, nyquistFreq, length(powerSpectrumAC)), powerSpectrumAC);
xlabel('Normalized frequency'), ylabel('Power [m^2 / s^2]');
title('Power spectrum');
grid minor;
subplot(4, 1, 3);
plot(linspace(0.0, nyquistFreq, length(powerSpectrumFull)), powerSpectrumFull);
xlabel('Normalized frequency'), ylabel('Power [m^2 / s^2]');
title('Spectrum of the full matrix');
grid minor;
subplot(4, 1, 4);
plot(linspace(0.0, nyquistFreq, length(powerSpectrumACFull)), powerSpectrumACFull);
xlabel('Normalized frequency'), ylabel('Power [m^2 / s^2]');
title('Power spectrum of the full matrix');
grid minor;

%% Spectrum of correlation matrices
clear all, close all, clc;

% data = load('C_DNS100_2Modes.mat'); 
% dt = 0.05;
data = load('C_DNS300_2Modes.mat');
dt = 0.25;
nyquistFreq = 1 / (2 * dt);
cov_v = data.c;
bt = data.bt;
clear data;

col_v = cov_v(:, 1);
row_v = cov_v(1, :);
col_fft = fftshift(fft(col_v));
col_fft = abs(col_fft(floor(length(col_fft) / 2) : end));
row_fft = fftshift(fft(row_v));
row_fft = abs(row_fft(floor(length(row_fft) / 2) : end));
diag_fft = fftshift(fft(diag(cov_v)));
diag_fft = abs(diag_fft(floor(length(diag_fft) / 2 ): end));
antidiag_fft = diag(fliplr(cov_v));
antidiag_fft = fftshift(fft(antidiag_fft));
antidiag_fft = abs(antidiag_fft(floor(length(antidiag_fft) / 2) : end));

figure, subplot(2, 2, 1);
plot(linspace(0.0, nyquistFreq, length(col_fft)), col_fft);
title('FFT of a column'), xlabel('Frequency [Hz]');
grid minor;
subplot(2, 2, 2);
plot(linspace(0.0, nyquistFreq, length(row_fft)), row_fft);
title('FFT of a row'), xlabel('Frequency [Hz]');
grid minor;
subplot(2, 2, 3);
plot(linspace(0.0, nyquistFreq, length(diag_fft)), diag_fft);
title('FFT of diagonal'), xlabel('Frequency [Hz]');
grid minor;
subplot(2, 2, 4);
plot(linspace(0.0, nyquistFreq, length(antidiag_fft)), antidiag_fft);
title('FFT of antidiagonal'), xlabel('Frequency [Hz]');
grid minor;

% Plot the 2D FFT in image
figure, subplot(2, 1, 1);
freqRange = linspace(-nyquistFreq, nyquistFreq, size(cov_v, 1));
imagesc(freqRange, freqRange, 100 * log( 1 + abs(fftshift(fft2(cov_v))))); colormap gray; title('FFT of c (log)');
axis xy;
subplot(2, 1, 2);
imagesc(freqRange, freqRange, abs(fftshift(fft2(cov_v)))); colormap gray; title('FFT of c (linear)')
axis xy;

% Mean of cols and rows
matrix_fft = abs(fftshift(fft2(cov_v)));
col_mean = mean(matrix_fft, 1);
col_mean = col_mean(length(col_mean) / 2 : end);
row_mean = mean(matrix_fft, 2);
row_mean = row_mean(length(row_mean) / 2 : end);

% Plot the mean of the ffts
figure, subplot(2, 1, 1);
plot(linspace(0.0, nyquistFreq, length(col_mean)), col_mean);
title('FFT of mean of columns'), xlabel('Frequency [Hz]');
grid minor;
subplot(2, 1, 2);
plot(linspace(0.0, nyquistFreq, length(row_mean)), row_mean);
title('FFT of mean of rows'), xlabel('Frequency [Hz]');
grid minor;

% Antidiag by another method
antidiag_fft2 = diag(fliplr(matrix_fft));
antidiag_fft2 = antidiag_fft2(length(antidiag_fft2) / 2 : end);

% Phase of the ffts
phase_antidiag = angle(fftshift(fft(diag(fliplr(cov_v)))));
phase_antidiag = phase_antidiag(length(phase_antidiag) / 2 : end);
phase_antidiag2 = angle(diag(fliplr(fftshift(fft2(cov_v)))));
phase_antidiag2 = phase_antidiag2(length(phase_antidiag2) / 2 : end);

figure, subplot(2, 2, 1), 
plot(linspace(0.0, nyquistFreq, length(antidiag_fft2)), antidiag_fft2);
title('Antidiagonal of fft2 matrix'), xlabel('Frequency [Hz]');
ylabel('Amplitude'), grid minor;
subplot(2, 2, 2);
plot(linspace(0.0, nyquistFreq, length(antidiag_fft)), antidiag_fft);
title('FFT of antidiagonal of matrix'), xlabel('Frequency [Hz]');
ylabel('Amplitude'), grid minor;
subplot(2, 2, 3);
plot(linspace(0.0, nyquistFreq, length(phase_antidiag2)), phase_antidiag2);
title('Antidiagonal of fft2 matrix'), xlabel('Frequency [Hz]');
ylabel('Phase'), grid minor;
subplot(2, 2, 4);
plot(linspace(0.0, nyquistFreq, length(phase_antidiag)), phase_antidiag);
title('FFT of antidiagonal of matrix'), xlabel('Frequency [Hz]');
ylabel('Phase'), grid minor;

%% Test each batch
clear all, close all, clc;

data = load('C_DNS100_16Modes.mat'); 
dt = 0.05;
% data = load('C_DNS300_4Modes.mat');
% dt = 0.25;
nyquistFreq = 1 / (2 * dt);
cov_v = data.c;
bt = data.bt;
clear data;

N = size(cov_v, 2);
% Estimate the small-scale velocity
cov_s = smallScaleVelocityCov(cov_v, bt);

% Estimate the period to know the size of the batches
period = periodicityFromAutocorrelation(cov_v);

% Compute the autocorrelation time in batches
autocorrelationTime = zeros(ceil(N / period), 1);
autocorrelation = cell(ceil(N / period), 1);
periodIndex = 1;
currBatch{1} = 1;
currPosition = 1;
while currBatch{1} ~= 0
    currBatch = nextMatrixPeriod(cov_s, period, currPosition, 'global');
    if currBatch{1} == 0
        break;
    end
    autocorrelation{periodIndex} = estimateAutocorrelation(currBatch);
    autocorrelationTime(periodIndex) = estimateAutocorrelationTime(currBatch);
    currPosition = currPosition + period;
    periodIndex = periodIndex + 1;
end

nBatch = 88;
figure, plot(1 : period, autocorrelation{nBatch})
title(['Correlation for period ' num2str(nBatch) ' (\tau_{corr} = ' num2str(autocorrelationTime(nBatch)) ')']);
xlabel('$\tau_{\mathrm{total}} \, \Delta t [samples / s]$', 'Interpreter', 'latex'), ylabel('Correlation');
grid minor;

% Estimate the different means
meanACTime = mean(autocorrelationTime)
weightedACTime = weightedACTimeMean(autocorrelationTime)


%% Test autocorrelation time in batches with synthetic data
%SET THE PERIOD TO N TO COMPARE TO THE CLASSIC METHOD
clear all, close all, clc;

T = 5.0;
N = 1000;
t_0 = 0.0;
t_f = 100.0;
durationT = t_f - t_0;
deltaT = (durationT / N);
t = linspace(t_0, t_f, N);
tau = 1.0;
sigma = 1 / (sqrt(2 * pi) * (tau / deltaT));

beta = @(x) 1 + 0.2 * cos(2 * pi * x / T);
alpha = @(x) 1 + 0.2 * sin(2 * pi * x / T);

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

actime = autocorrelationTimeInBatches(c, bt_test, 'global') / (N / durationT);
actimeOld = autocorr_time(c, bt_test) / (N / durationT);

% Estimate the means
actimeMean = mean(actime);
% actimeWeighted = weightedACTimeMean(actime);

% Estimate the error to the reference
errorMean = abs(actimeMean - actimeOld)
error = sqrt(abs(actimeMean.^2 / (2 * pi) - tau.^2)) / durationT
errorOld = sqrt(abs(actimeOld.^2 / (2 * pi) - tau.^2)) / durationT

%% Test autocorrelation time in batches for heterogeinities
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

% beta = @(x, beta0) 1 + beta0 * cos(2 * pi * x / T);
beta = @(x) 1 + 0.2 * cos(2 * pi * x / T);
alpha = @(x, alpha0) 1 + alpha0 * sin(2 * pi * x / T);
% alpha = @(x) 1 + 0.2 * sin(2 * pi * x / T);
cov_x = @(tau, dt, st, alpha0) sigma.^2 * alpha(st, alpha0) * ...
    exp(-0.5 * (dt / tau / beta(st) ).^2);

nVar = 100;
% betaVar = linspace(0.0, 100, nVar);
alphaVar = linspace(0.0, 100, nVar);

for k = 1 : nVar
    k
    for i = 1 : N
        for j = 1 : N
            dt = abs(i - j) * deltaT;
            st = (i + j) * deltaT;
            c(i, j) = cov_x(tau, dt, st, alphaVar(k));
        end
    end
    bt_test = zeros(N, 1); % Disregard the computation of the large-scale velocity
        
    actime = autocorrelationTimeInBatches(c, bt_test, 'global') / (N / durationT);
    actimeMean(k) = mean(actime);
    actimeOld(k) = autocorr_time(c, bt_test) / (N / durationT);
end

% Estimate the error to the reference
error = sqrt(abs(actimeMean.^2 / (2 * pi) - tau.^2)) / durationT;
errorOld = sqrt(abs(actimeOld.^2 / (2 * pi) - tau.^2)) / durationT;

% Plot the error surfaces
figure, hold on, plot(actimeMean / sqrt(2 * pi)), plot(tau * ones(size(actimeMean)));
figure, hold on, plot(actimeOld / sqrt(2 * pi)), plot(tau * ones(size(actimeOld)));
figure, plot(error), title('Error of \tau in batches');
figure, plot(errorOld), title('Error of \tau normal');
