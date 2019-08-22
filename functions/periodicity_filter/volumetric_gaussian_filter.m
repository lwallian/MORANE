function [filtered_field] = volumetric_gaussian_filter(field, MX)
%VOLUMETRIC_GAUSSIAN_FILTER Filters with a 3x3x3 gaussian kernel a
%volumetric field in each of its dimensions
%   @param field: input field to filter [Mx*My*(Mz), T, d]
%   @param MX: grid details for the filter [Mx, My, (Mz)]
%
% Author: Agustin PICARD, intern @ Scalian with Valentin RESSEGUIER as
% supervisor
%

[M, T, d] = size(field);

if d == 3
    kernel = repmat([1 / 16, 1 / 8, 1 / 16;
                1 / 8, 1 / 4, 1 / 8;
                1 / 16, 1 / 8, 1 / 16], [1, 1, 3]);
else
    kernel = [1 / 16, 1 / 8, 1 / 16;
                1 / 8, 1 / 4, 1 / 8;
                1 / 16, 1 / 8, 1 / 16];
end

field = reshape(field, [MX, T, d]);
filtered_field = zeros(size(field));

if d == 2
    for t = 1 : T
        for i = 2 : MX(1) - 1
            for j = 2 : MX(2) - 1
                filtered_field(i, j, t, :) = volumetric_convolution(field(i - 1 : i + 1, j - 1 : j + 1, t, :), d);
            end
        end
    end
else
    for t = 1 : T
        for i = 2 : MX(1) - 1
            for j = 2 : MX(2) - 1
                for k = 2 : MX(3) - 1
                    filtered_field(i, j, k, t, :) = volumetric_convolution(field(i - 1 : i + 1, j - 1 : j + 1, k - 1 : k + 1, t, :), d);
                end
            end
        end
    end
end

filtered_field = reshape(filtered_field, [M, T, d]);

end


function filtered_points = volumetric_convolution(points, kernel, d)

persistent colons;
if isempty(colons)
    colons = generate_colons(d);
end

filtered_points = zeros(d, 1);

for k = 1 : d
    filtered_points(k) = eval(["sum(points(", colons, num2str(k), ") .* kernel, 'all');"]);
end

end


function c = generate_colons(n)

c = repmat(':,', [n, 1]);

end


function quantile = gaussian_quantile(p)

quantile = sqrt(2) * erfcinv(2 * p);

end

