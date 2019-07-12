function [param] = simpleMA_field_filter(param, window_size)
%SIMPLEMA_FIELD_FILTER Filters the velocity field in time with a moving
%average filter with the specified window size
%   @param param: structure containing the following elements :
%   folder_data, type_data, N_tot, M, MX, d
%   @param window_size: size of the filter's window
%
% Author: Agustin PICARD, intern @ Scalian with Valentin RESSEGUIER as
% supervisor
%

assert(isinteger(window_size), 'Window size must be an integer');
assert(isfield(param, 'folder_data'), 'Missing folder_data field in parameter structure');
assert(isfield(param, 'type_data'), 'Missing type_data field in parameter structure');
assert(isfield(param, 'N_tot'), 'Missing N_tot field in parameter structure');
assert(isfield(param, 'M'), 'Missing M field in parameter structure');
assert(isfield(param, 'MX'), 'Missing MX field in parameter structure');
assert(isfield(param, 'd'), 'Missing d field in parameter structure');


% dt = param.dt;
N_tot = param.N_tot;
% m = param.nb_modes;
% nu = param.viscosity;
% T = N_tot*dt;
M = param.M;
% dX = param.dX;
MX = param.MX;
d = param.d;

if ~param.data_in_blocks.bool
    % Deal first with the case where the data is in just one file
    name_file_U_centered=[param.folder_data, param.type_data, '_U_centered'];
    load(name_file_U_centered, 'U');
    filtered_field = MA_filter_one_block_field(U, MX, d, window_size);
    filename = get_save_filename(param);
    save(filename, 'filtered_field');
else
    % Load the first file to get some statistics to manage the amount of
    % files to load per time step
    filename = [param.folder_data param.type_data num2str(1) '_U_centered'];
    load(filename, 'U');
    field = U; % Rename U in case we'll need to load multiple files
    clear U;
    field = reshape(field, M, [], d);
    data_size = size(field); % [M, t, d]
    T_file = data_size(2); % amount of time steps per file
    % Simplified case, otherwise it would be quite difficult to manage.
    % TODO: code it for the general case
    if window_size > T_file
        window_size = T_file;
        warning('Window size too big. Fixing it to temporal file length')
    end
    t = 1; % global time
    n_file = 1; % next file to read
    
    % Do the first step that is going to differ from the center and last
    % ones
    filename = get_field_filename(param, n_file);
    field = get_next_field(filename, field, M, d, window_size);
    n_file = n_file + 1;
    filtered_field = MA_filter_block(field, MX, d, window_size, 'first');
    filename = get_save_filename(param, n_file);
    U = filtered_field;
    save(filename, 'U');
    clear U filtered_field;
    t = t + T_file;
    
    % Filter the rest of the field
    while t < N_tot - T_file
        filename = get_field_filename(param, n_file);
        field = get_next_field(filename, field, M, d, window_size);
        n_file = n_file + 1;
        filtered_field = MA_filter_block(field, MX, d, window_size, 'center');
        filename = get_save_filename(param, n_file);
        U = filtered_field;
        save(filename, 'U');
        clear U filtered_field;
        t = t + T_file;
    end
    
    filename = get_field_filename(param, n_file);
    field = get_next_field(filename, field, M, d, window_size);
    n_file = n_file + 1;
    filtered_field = MA_filter_block(field, MX, d, window_size, 'last');
    filename = get_save_filename(param, n_file);
    U = filtered_field;
    save(filename, 'U');
    clear U filtered_field;
    t = t + T_file;
end

param.filtered.filename = get_save_filename(param);
param.filtered.order = order;
param.filtered.delay = delay;
param.filtered.n_blocks = n_file;

end


function [field] = get_next_field(filename, field, M, d, window_size)

load(filename, 'U');
U = reshape(U, M, [], d);
field = cat(2, field(:, end - window_size / 2 - 1 : end, :), U);
clear U;
    
end


function [load_filename] = get_field_filename(param, n_block)

if nargin == 2
    load_filename = [param.folder_data, param.type_data, num2str(n_block), '_U_centered'];
else
    load_filename = [param.folder_data, param.type_data, '_U_centered'];
end

end


function [filtered_field] = MA_filter_block(field, MX, d, window_size, mode)

assert(strcmp(mode, 'first') || strcmp(mode, 'last') || strcmp(mode, 'center'))

[~, T, ~] = size(field);
field = reshape(field, [MX, T, d]);
filtered_field = zeros(size(field));

if strcmp(mode, 'first')
    if d == 2
        for l = 1 : d
            for i = 1 : MX(1)
                for j = 1 : MX(2)
                    for t = 1 : (T - window_size / 2)
                        if t < window_size / 2
                            aux_field = [zeros(window_size / 2 - t), field(i, j, l, 1 : t + window_size / 2)];
                            filtered_field(i, j, l, t) = mean(aux_field);
                        else
                            filtered_field(i, j, l, :) = mean(field(i, j, l, t - window_size / 2 : t + window_size / 2));
                        end
                    end
                end
            end
        end
    else
        for l = 1 : d
            for i = 1 : MX(1)
                for j = 1 : MX(2)
                    for k = 1 : MX(3)
                        for t = 1 : (T - window_size / 2)
                            if t < window_size / 2
                                aux_field = [zeros(window_size / 2 - t), field(i, j, k, l, 1 : t + window_size / 2)];
                                filtered_field(i, j, k, l, t) = mean(aux_field);
                            else
                                filtered_field(i, j, k, l, :) = mean(field(i, j, k, l, t - window_size / 2 : t + window_size / 2));
                            end
                        end
                    end
                end
            end
        end
    end
elseif strcmp(mode, 'center')
    if d == 2
        for l = 1 : d
            for i = 1 : MX(1)
                for j = 1 : MX(2)
                    for t = window_size / 2 : (T - window_size / 2)
                        filtered_field(i, j, l, :) = mean(field(i, j, l, t - window_size / 2 : t + window_size / 2));
                    end
                end
            end
        end
    else
        for l = 1 : d
            for i = 1 : MX(1)
                for j = 1 : MX(2)
                    for k = 1 : MX(3)
                        for t = window_size / 2 : (T - window_size / 2)
                            filtered_field(i, j, k, l, :) = mean(field(i, j, k, l, t - window_size / 2 : t + window_size / 2));
                        end
                    end
                end
            end
        end
    end
else
    if d == 2
        for l = 1 : d
            for i = 1 : MX(1)
                for j = 1 : MX(2)
                    for t = 1 : (T - window_size / 2)
                        if t > T
                            aux_field = [field(i, j, l, t - window_size / 2 : end), zeros(T - window_size / 2 - t)];
                            filtered_field(i, j, l, t) = mean(aux_field);
                        else
                            filtered_field(i, j, l, :) = mean(field(i, j, l, t - window_size / 2 : t + window_size / 2));
                        end
                    end
                end
            end
        end
    else
        for l = 1 : d
            for i = 1 : MX(1)
                for j = 1 : MX(2)
                    for k = 1 : MX(3)
                        for t = 1 : T
                            if t > T
                                aux_field = [field(i, j, k, l, t - window_size / 2 : end), zeros(T - window_size / 2 - t)];
                                filtered_field(i, j, k, l, t) = mean(aux_field);
                            else
                                filtered_field(i, j, k, l, :) = mean(field(i, j, k, l, t - window_size / 2 : t + window_size / 2));
                            end
                        end
                    end
                end
            end
        end
    end
end

filtered_field = reshape(filtered_field, M, [], d);

end


function [filtered_field] = MA_filter_one_block_field(field, MX, d, window_size)

[~, T, ~] = size(field);
field = reshape(field, [MX, T, d]);
filtered_field = zeros(size(field));

if d == 2
    for l = 1 : d
        for i = 1 : MX(1)
            for j = 1 : MX(2)
                for t = 1 : T
                    if t < window_size / 2
                        aux_field = [zeros(window_size / 2 - t), field(i, j, l, 1 : t + window_size / 2)];
                        filtered_field(i, j, l, t) = mean(aux_field);
                    elseif t > T - window_size / 2
                        aux_field = [field(i, j, l, t - window_size / 2 : end), zeros(T - window_size / 2 - t)];
                        filtered_field(i, j, l, t) = mean(aux_field);
                    else
                        filtered_field(i, j, l, :) = mean(field(i, j, l, t - window_size / 2 : t + window_size / 2));
                    end
                end
            end
        end
    end
else
    for l = 1 : d
        for i = 1 : MX(1)
            for j = 1 : MX(2)
                for k = 1 : MX(3)
                    for t = 1 : T
                        if t < window_size / 2
                            aux_field = [zeros(window_size / 2 - t), field(i, j, k, l, 1 : t + window_size / 2)];
                            filtered_field(i, j, k, l, t) = mean(aux_field);
                        elseif t > T - window_size / 2
                            aux_field = [field(i, j, k, l, t - window_size / 2 : end), zeros(T - window_size / 2 - t)];
                            filtered_field(i, j, k, l, t) = mean(aux_field);
                        else
                            filtered_field(i, j, k, l, :) = mean(field(i, j, k, l, t - window_size / 2 : t + window_size / 2));
                        end
                    end
                end
            end
        end
    end
end

filtered_field = reshape(filtered_field, M, [], d);

end


function [save_filename] = get_save_filename(param, n_block)

if nargin == 2
    save_filename = [param.folder_data, param.type_data, num2str(n_block), '_MA'];
else
    save_filename = [param.folder_data, param.type_data, '_MA'];
end

end

