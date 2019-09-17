function [spect] = fxSpectrogram(inputArray)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

if isnumeric(inputArray)
    [M, N] = size(inputArray);
    
    fftArray = zeros(M, N);
    for i = 1 : N
        aux = fftshift(fft([inputArray(:, i)', zeros(M, 1)']));
        fftArray(:, i) = abs(aux(length(aux) / 2 : end));
    end
elseif iscell(inputArray)
    N = length(inputArray);
    M = length(inputArray{1});
    
    if length(inputArray{N}) ~= M
        fftArray = zeros(M, N - 1);
        for i = 1 : N - 1
            aux = fftshift(fft([inputArray{i}', zeros(M, 1)']))';
            aux = abs(aux(M + 1 : end));
            fftArray(:, i) = aux;
        end
    else
        fftArray = zeros(M, N);
        for i = 1 : N
            aux = fftshift(fft([inputArray{i}', zeros(M, 1)']));
            fftArray(:, i) = abs(aux(M + 1 : end));
        end
    end
end

xAxis = linspace(0.0, 1.0, size(fftArray, 1));
yAxis = linspace(0.0, 1.0, size(fftArray, 2));
figure, imagesc(xAxis, yAxis, fftArray), axis xy;
xlabel('Normalized frequency'), ylabel('Normalized time');

spect = fttArray;

end

