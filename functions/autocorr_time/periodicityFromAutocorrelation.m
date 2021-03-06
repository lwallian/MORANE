function [period] = periodicityFromAutocorrelation(cov_s)
%PERIODICITYFROMAUTOCORRELATION Estimates the covariance's period
%   @param cov_s: small-scale velocity's covariance matrix
%   @return: covariance's most important's mode period.
%
% Author: Agustin PICARD, intern @ Scalian with Valentin RESSEGUIER as
% supervisor
%
N = size(cov_s, 2);

% Estimate the autocorrelation function
autocorrFunction = estimateAutocorrelation(cov_s);

% Calculate the fft and estimate the period as the biggest mode
power_spectrum = fftshift(fft([autocorrFunction', zeros(size(autocorrFunction'))]));
power_spectrum = abs(power_spectrum(floor(length(power_spectrum) / 2) : end)); % just keep half of the amplitude

% The period corresponds to the central frequency
[~, frequence_central] = max(power_spectrum);
period = ceil(2.0 * length(power_spectrum) / frequence_central);

end

