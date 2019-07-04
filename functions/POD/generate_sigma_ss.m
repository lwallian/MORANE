function [sigma_ss] = generate_sigma_ss(f, dims, stdev, dx)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

d = length(dims);
sigma_ss = zeros(dims);

if d == 2
    c = [dims(1) / 2, dims(2) / 2] .* dx;
    x = 1 : dx : dims(1);
    x = x - c(1);
    y = 1 : dx : dims(2);
    y = y - c(2);
    for i = 1 : dims(1)
        for j = 1 : dims(2)
            sigma_ss(i, j) = sin(2 * pi * f * norm([x(i), y(j)])) * ...
                exp(-norm([x(i), y(j)]).^2 / stdev.^2);
        end
    end
elseif d == 3
    c = [dims(1) / 2, dims(2) / 2, dims(3) / 2];
    x = 1 : dx : dims(1);
    x = x - c(1);
    y = 1 : dx : dims(2);
    y = y - c(2);
    z = 1 : dx : dims(3);
    z = z - c(3);
    for i = 1 : dims(1)
        for j = 1 : dims(2)
            for k = 1 : dims(3)
                sigma_ss(i, j, k) = sin(2 * pi * f * norm([x(i), y(j), z(k)])) * ...
                    exp(-norm([x(i), y(j), z(k)]).^2 / stdev.^2);
            end
        end
    end
else
    error('Invalid sigma_ss dimensions')
end

end