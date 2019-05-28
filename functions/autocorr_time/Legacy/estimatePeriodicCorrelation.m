function [periodicity] = estimatePeriodicCorrelation(covMatrix)
%ESTIMATEPERIODICPART Estimates the periodic part of the correlation
%function.
%   @param covMatrix: covariance matrix
%   @return: one period of the periodic part of the correlation function
%
% Author: Agustin PICARD, intern @ Scalian with Valentin RESSEGUIER as
% supervisor
%

% Estimate the matrix's periodicity
period = periodicityFromAutocorrelation(covMatrix);

periodicity = zeros(period, 1);
periodIndex = 1;
currBatch{1} = 1;
currPosition = 1;
while currBatch{1} ~= 0
    currBatch = nextMatrixPeriod(covMatrix, period, currPosition, 'global');
    if currBatch{1} == 0
        break;
    elseif length(currBatch) < period + 1
        break;
    end
    periodicity = periodicity + estimateAutocorrelation(currBatch);
    currPosition = currPosition + period;
    periodIndex = periodIndex + 1;
end

periodicity = periodicity / (periodIndex - 1);

end

