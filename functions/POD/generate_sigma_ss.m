function [sigma_ss] = generate_sigma_ss(f, dims, stdev, dX)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

d = length(dX);
sigma_ss = zeros(dims);

if d == 2
    x = 1 : dims(1);
    y = 1 : dims(2);
    c = [dims(1) / 2, dims(2) / 2];
    x = x - c(1);
    y = y - c(2);
    for l = 1 : d
        for i = 1 : length(x)
            for j = 1 : length(y)
                sigma_ss(i, j, l) = sin(2 * pi * f * norm([x(i), y(j)])) * ...
                    exp(-norm([x(i), y(j)]).^2 / stdev.^2);
            end
        end
    end
elseif d == 3
    x = 1 : dims(1);
    y = 1 : dims(2);
    z = 1 : dims(3);
    c = [dims(1) / 2, dims(2) / 2, dims(3) / 2];
    x = x - c(1);
    y = y - c(2);
    z = z - c(3);
    for l = 1 : d
        for i = 1 : length(x)
            for j = 1 : length(y)
                for k = 1 : length(z)
                    sigma_ss(i, j, k, l) = sin(2 * pi * f * norm([x(i), y(j), z(k)])) * ...
                        exp(-norm([x(i), y(j), z(k)]).^2 / stdev.^2);
                end
            end
        end
    end
else
    error('Invalid sigma_ss dimensions')
end

end