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
xlabel('Period index'), ylabel('\tau_{corr} [samples / s]');

% Estimate the different means
meanACTime = mean(autocorrTime)
weightedACTime = weightedACTimeMean(autocorrTime)

%% Comparison of the spectrum of the autocorr and the cov matrix
clear all, close all, clc;

data = load('C_DNS100_2Modes.mat');
% data = load('C_DNS300_4Modes.mat');
cov_v = data.c;
bt = data.bt;
clear data;

% Calculate the small-scale velocity field
N = size(cov_v, 2);
cov_s = smallScaleVelocityCov(cov_v, bt);
clear cov_v;

% Estimate the autocorrelation function
autocorrFunction = estimateAutocorrelation(cov_s);

% Calculate the fft and estimate the period as the biggest mode
powerSpectrumAC = fftshift(fft(autocorrFunction));
powerSpectrumAC = abs(powerSpectrumAC(floor(length(powerSpectrumAC) / 2) : end)); % just keep half of the amplitude

[~, frequenceCentral] = max(powerSpectrumAC);
periodAC = ceil(N / frequenceCentral);

% Now for the spectrum of the covariance matrix
% Get the antidiagonal
antiDiag = diag(fliplr(cov_s));

% Calculate its spectrum
powerSpectrumCov = fftshift(fft(antiDiag));
powerSpectrumCov = abs(powerSpectrumCov(floor(length(powerSpectrumCov) / 2) : end)); % just keep half of the amplitude

% Get the central frequency
[~, frequence_central] = max(powerSpectrumCov);
periodCov = ceil(N / frequence_central);

% Plot both in subplots
figure, subplot(2, 1, 1);
plot(linspace(0.0, 1.0, length(powerSpectrumCov)), powerSpectrumCov);
xlabel('Normalized frequency'), ylabel('Power [m^2 / s^2]');
title('Spectrum of covariance matrix (DNS300 - 4 modes)');
grid minor;
subplot(2, 1, 2);
plot(linspace(0.0, 1.0, length(powerSpectrumAC)), powerSpectrumAC);
xlabel('Normalized frequency'), ylabel('Power [m^2 / s^2]');
title('Power spectrum');
grid minor;

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

beta = @(x) 1 + 0.5 * cos(2 * pi * x / T);
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
