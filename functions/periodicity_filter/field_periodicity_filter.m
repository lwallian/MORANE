function [param] = field_periodicity_filter(param, delay, order)
%FIELD_PERIODICITY_FILTER Filters out the periodic component close to the given delay of a 
%velocity field with an LMS adaptive filter of a given order
%   @param param: structure containing the following elements :
%   folder_data, type_data, N_tot, M, MX, d
%   @param order: amount of taps in the LMS filter estimated during the
%   correlation function filtering process
%   @param delay: delay that is going to be introduced to the field to
%   suppress that periodic component
%   @return param: modified parameter structure with the filenames of each
%   part of the filtered field
%
% Author: Agustin PICARD, intern @ Scalian with Valentin RESSEGUIER as
% supervisor
%

assert(isinteger(delay), 'Delay must be an integer');
assert(isinteger(order), 'Filter order must be an integer');
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
    name_file_U_centered=[param.folder_data, param.type_data, '_U_centered'];
    load(name_file_U_centered, 'U');
    filtered_field = LMS_filter_one_block_field(U, MX, d, delay, order);
    filename = get_save_filename(param);
    save(filename, 'filtered_field');
else
    % Initialize the filter with zeros
	w_filter = zeros([M, d, order]);
    
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
    if order > T_file
        order = T_file;
        warning('Order too big. Fixing it to temporal file length')
    end
    t_delay = delay; % amount of steps left to delay the input signal
    t = 1; % global time
    n_file = 1; % next file to read
    reg_term = 1e-4 * order * max(field(:,1,:), [], 'all');
    if T_file > delay
        % We'll work with two files max
        % Do the first delayed input by padding
        ref_signal = [zeros([MX, delay, d]), field(:, delay : end, :)];
        ref_signal = ref_signal(1 : T_file);
        t_delay = 0;
        n_file = n_file + 1;
        [filtered_field, w_filter] = LMS_filter_block(field, ref_signal, w_filter, d, order, reg_term);
        filename = get_save_filename(param, n_file);
        U = filtered_field;
        save(filename, 'U');
        clear U;
        t = t + T_file;
        
        % Filter the rest of the field
        while t < N_tot
            filename = get_field_filename(param, n_file);
            [field, ref_signal] = get_next_field(filename, field, M, d, order);
            [filtered_field, w_filter] = LMS_filter_block(field, ref_signal, w_filter, d, order, reg_term);
            filename = get_save_filename(param, n_file);
            U = filtered_field;
            save(filename, 'U');
            clear U filtered_field;
            t = t + T_file;
        end
    else
        n_delayed = 0; % Amount of files we delay the reference signal
        % If the delay is more than a file, the filter is not going to
        % change the field's value, so we just save the original field
        while t_delay > T_file
            filename = get_save_filename(param, n_file);
            U = field;
            save(filename, 'U');
            clear U;
            n_file = n_file + 1;
            t_delay = t_delay - T_file;
            t = t + T_file;
            filename = get_field_filename(param, n_file);
            load(filename, 'U');
            field = U;
            clear U;
            n_delayed = n_delayed + 1;
        end
        t_delay = 0;
        filename = get_field_filename(param, n_file - n_delayed);
        ref_signal = get_next_reference(filename, [], delay);
        n_file = n_file + 1;
        [filtered_field, w_filter] = LMS_filter_block(field, ref_signal, w_filter, d, order, reg_term);
        filename = get_save_filename(param, n_file);
        U = filtered_field;
        save(filename, 'U');
        clear U filtered_field;
        t = t + T_file;
        
        % Filter the rest of the field
        while t < N_tot
            filename = get_field_filename(param, n_file - n_delayed);
            ref_signal = get_next_reference(filename, ref_signal, delay);
            filename = get_field_filename(param, n_file);
            field = get_next_field(filename, field, M, d, order);
            n_file = n_file + 1;
            [filtered_field, w_filter] = LMS_filter_block(field, ref_signal, w_filter, d, order, reg_term);
            filename = get_save_filename(param, n_file);
            U = filtered_field;
            save(filename, 'U');
            clear U filtered_field;
            t = t + T_file;
        end
    end
end

param.filtered.filename = get_save_filename(param);
param.filtered.order = order;
param.filtered.delay = delay;
param.filtered.n_blocks = n_file;

end


function [field, reference] = get_next_field(filename, field, M, d, order, reference, delay)

if nargin == 7
    load(filename, 'U');
    U = reshape(U, M, [], d);
    T_U = size(U);
    if T_U(2) > delay
        reference = cat(2, field(:, end - delay : end, :), U(:, 1 : end - delay, :));
    else
        reference = cat(2, field(:, end - delay : end, :), U(:, :, :));
    end
    field = cat(2, field(:, end - order - 1 : end, :), U);
    clear U;
else
    load(filename, 'U');
    U = reshape(U, M, [], d);
    field = cat(2, field(:, end - order - 1 : end, :), U);
    clear U;
end

end


function [ref_signal] = get_next_reference(filename, ref_signal, delay)

load(filename, 'U');
U = reshape(U, M, [], d);
T_U = size(U);
if isempty(ref_signal)
    ref_signal = [zeros(M, delay, d), U(:, 1 : end - delay, :)];
else
    if T_U(2) > delay
        ref_signal = cat(2, ref_signal(:, end - delay : end, :), U(:, 1 : end - delay, :));
    else
        ref_signal = cat(2, ref_signal(:, end - delay : end, :), U(:, :, :));
    end
end
clear U;

end


function [load_filename] = get_field_filename(param, n_block)

if nargin == 2
    load_filename = [param.folder_data, param.type_data, num2str(n_block), '_U_centered'];
else
    load_filename = [param.folder_data, param.type_data, '_U_centered'];
end

end


function [filtered_field, w_filter] = LMS_filter_block(field, reference, w_filter, d, order, reg_term)

filtered_field = zeros(size(reference));

[M, N, ~] = size(reference);

for i = 1 : N - order - 1
    for k = 1 : d
        % For the x part
        filtered_field(1 : d : end, i, k) = field(1 : d : end, i, k) ...
            - w_filter(1 : d : end, k, :)' * reference(1 : d : end, i : i + order - 1, k);
        w_filter(1 : d : end, k, :) = w_filter(1 : d : end, k, :) + ...
            (2 / norm(reference(1 : d : end, i : i + order - 1, k)).^2 ...
            + reg_term) * filtered_signal(1 : d : M, i, k) ...
            * reference(1 : d : end, i : i + order - 1, k);
        % For the y part
        filtered_field(2 : d : end, i, k) = field(2 : d : end, i, k) ...
            - w_filter(2 : d : end, k, :)' * reference(2 : d : end, i : i + order - 1, k);
        w_filter(2 : d : end, k, :) = w_filter(2 : d : end, k, :) + ...
            (2 / norm(reference(2 : d : end, i : i + order - 1, k)).^2 ...
            + reg_term) * filtered_signal(2 : d : M, i, k) ...
            * reference(2 : d : end, i : i + order - 1, k);
        if d == 3
            % For the z part
            filtered_field(3 : d : end, i, k) = field(3 : d : end, i, k) ...
                - w_filter(3 : d : end, k, :)' * reference(3 : d : end, i : i + order - 1, k);
            w_filter(3 : d : end, k, :) = w_filter(3 : d : end, k, :) + ...
                (2 / norm(reference(3 : d : end, i : i + order - 1, k)).^2 ...
                + reg_term) * filtered_signal(3 : d : M, i, k) ...
                * reference(3 : d : end, i : i + order - 1, k);
        end
    end
end
    

end


function [filtered_field] = LMS_filter_one_block_field(field, MX, d, delay, order)

[~, T, ~] = size(field);
field = reshape(field, [MX, T, d]);
filtered_field = zeros(size(field));

if d == 2 
    for l = 1 : d
        for i = 1 : MX(1)
            for j = 1 : MX(2)
                filtered_field(i, j, l, :) = LMSFilter(field(i, j, l, :), order, delay);
            end
        end
    end
else
    for l = 1 : d
        for i = 1 : MX(1)
            for j = 1 : MX(2)
                for k = 1 : MX(3)
                    filtered_field(i, j, k, l, :) = LMSFilter(field(i, j, k, l, :), order, delay);
                end
            end
        end
    end
end

filtered_field = reshape(filtered_field, M, [], d);

end

function [save_filename] = get_save_filename(param, n_block)

if nargin == 2
    save_filename = [param.folder_data, param.type_data, num2str(n_block), '_filtered'];
else
    save_filename = [param.folder_data, param.type_data, '_filtered'];
end

end
