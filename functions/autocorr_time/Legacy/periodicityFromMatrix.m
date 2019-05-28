function [period] = periodicityFromMatrix(m)
%PERIODICITYFROMMATRIX Calculates the characteristic period of the input
%matrix given the fact that it's going to be on the antidiagonals
%   @param m: input matrix
%   @return: period of its largest mode
%
% Author: Agustin PICARD, intern @ Scalian with Valentin RESSEGUIER as
% supervisor
%
N = size(m, 2);

% Get the antidiagonal
antiDiag = diag(fliplr(m));

% Calculate its spectrum
power_spectrum = fftshift(fft([antiDiag', zeros(size(antiDiag'))]));
power_spectrum = abs(power_spectrum(floor(length(power_spectrum) / 2) : end)); % just keep half of the amplitude

% Get the central frequency
[~, frequence_central] = max(power_spectrum);
period = ceil(length(power_spectrum) / frequence_central);

end

