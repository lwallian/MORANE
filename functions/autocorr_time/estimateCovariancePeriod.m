function [period] = estimateCovariancePeriod(cov_s)
%ESTIMATECOVARIANCEPERIOD Estimates the covariance's period
%   @param cov_s: small-scale velocity's covariance matrix
%
% Author: Agustin PICARD, intern @ Scalian with Valentin RESSEGUIER as
% supervisor
%
N = size(cov_s, 2);

% Estimate the autocorrelation function
autocorrFunction = estimateAutocorrelation(cov_s);

% Calculate the fft and estimate the period as the biggest mode
power_spectrum = fftshift(fft(autocorrFunction));
power_spectrum = abs(power_spectrum(floor(N / 2) : end)); % just keep half of the amplitude

[~, frequence_central] = max(power_spectrum);
period = ceil(2 * N / frequence_central);

end

