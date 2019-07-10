function [param] = field_periodicity_filter(param, delay, order)
%UNTITLED2 Summary of this function goes here
%   @param order: amount of taps in the LMS filter estimated during the
%   correlation function filtering process
%   @param delay: delay that is going to be introduced to the field to
%   suppress that periodic component
%   @return param: modified parameter structure with the filenames of each
%   part of the filtered field

dt = param.dt;
N_tot = param.N_tot;
m = param.nb_modes;
nu = param.viscosity;
T = N_tot*dt;
M = param.M;
dX = param.dX;
MX = param.MX;
d = param.d;
lambda = param.lambda;

if param.data_in_blocks.bool
    name_file_U_centered=[param.folder_data param.type_data '_U_centered'];
    load(name_file_U_centered);
    filtered_field = LMS_filter_one_block_field(U, MX, d, delay, order);
    filename = get_filename(param);
    save(filename, 'filtered_field');
elseif ~param.data_in_blocks && ~param.big_data
    % TODO: process each data point on the grid as a time series iteratively
	w_filter = zeros([M, d, order]);
    
    % Load the first file to get some statistics to manage the amount of
    % files to load per time step
    filename = [param.folder_data param.type_data num2str(1) '_U_centered'];
    load(filename, 'U');
    field = U; % Rename U in case we'll need to load multiple files
    clear U;
    data_size = size(field); % [M, t, d]
    T_file = data_size(2);
    t = 1;
    if T_file > delay
        % We'll work with two files max
        % Do the first delayed input by padding
        ref_signal = [zeros([MX, delay, d]), field(:, delay : end, :)];
        ref_signal = ref_signal(1 : T_file);
    else
        ref_signal = zeros([MX, delay, d]); % plutôt travailler avec ref à taux fixe
        
        n_files = ceil(delay / T_file); % Amount of files to load
        for i = 2 : n_files
            filename = [param.folder_data, param.type_data, num2str(i), '_U_centered'];
            load(filename, 'U');
            field = cat(2, field, U);
            clear U;
        end
    end
        
            
else
	% TODO: Deal with the multiple files problem, namely having to open
    % multiple ones at the same time if the order or the delay are too big
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


end

function [save_filename] = get_filename(param, n_block)

if nargin == 2
    save_filename = [param.folder_data, param.type_data, num2str(n_block)];
else
    save_filename = [param.folder_data, param.type_data];
end

end
