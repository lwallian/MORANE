function [aperiodic] = estimateAperiodicCorrelation(covMatrix, lag, periodicPart)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
N = size(covMatrix, 2);

% Estimate the periodic part of the correlation function
% periodicPart = estimatePeriodicCorrelation(covMatrix);
period = length(periodicPart);

% Estimate the correlation function and then substract its periodic part
aperiodic = zeros(lag, 1);
periodIndex = 1;
currBatch{1} = 1;
currPosition = 1;
while currBatch{1} ~= 0
    if currPosition + period >= lag
        currBatch = nextMatrixPeriod(covMatrix, lag - currPosition, currPosition, 'global');
        aperiodic(currPosition : lag - 1) = estimateAutocorrelation(currBatch) - periodicPart(1 : (lag - currPosition));
        break;
    else
        currBatch = nextMatrixPeriod(covMatrix, period, currPosition, 'global');
    end
    if currBatch{1} == 0
        break;
    elseif length(currBatch) < period + 1
        break;
%         aperiodic(currPosition : currPosition + length(currBatch)) = estimateAutocorrelation(currBatch) - periodicPart(1 : end - (period - length(currBatch) + 1));
    end
    aperiodic(currPosition : currPosition + period - 1) = estimateAutocorrelation(currBatch) - periodicPart;
    currPosition = currPosition + period;
    periodIndex = periodIndex + 1;
end
    

end

