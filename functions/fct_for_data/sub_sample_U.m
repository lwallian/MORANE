function param = sub_sample_U(param)
% Subsample in time the residual velocity field
% contained in temporary files
%

%% Initialization
param.folder_file_U_temp = fct_folder_temp(param);

% Subsampling rate
n_subsampl_decor = param.decor_by_subsampl.n_subsampl_decor;
t_local = 1; % index of the snapshot in a file before subsampling
t_local_sub = 1; % index of the snapshot in a file after subsampling

if param.data_in_blocks.bool % if data are saved in several files
    big_T = 1; % index of the file
    param_ref.name_file_U_temp={};
    name_file_U_centered=[param.folder_data param.type_data ...
        num2str(big_T) '_U_centered'];
else
    param.data_in_blocks.len_blocks = param.N_tot;
    name_file_U_centered = [param.folder_data param.type_data ...
                             '_U_centered'];
end

if ~ (param.big_data || param.data_in_blocks.bool)
    %% Load
    load(name_file_U_centered, 'U');
    % Subsample
    U = U(:, 1 : n_subsampl_decor : end, :);
    %% Save
    param.name_file_U_temp = [param.folder_file_U_temp param.type_data '_U_temp'];
    save(param.name_file_U_temp, 'U', '-v7.3');
    clear U
elseif ~ param.big_data
    vt = zeros(1, param.N_tot);
    vt(1 : n_subsampl_decor : end) = true;
    vt = reshape(vt, [param.data_in_blocks.len_blocks ...
        param.data_in_blocks.nb_blocks]);
    param.data_in_blocks.mat_subsample = vt;
    for big_T = 1 : param.data_in_blocks.nb_blocks % loop on files
        name_file_U_centered = [param.folder_data param.type_data ...
            num2str(big_T) '_U_centered'];
        % Load new file
        load(name_file_U_centered, 'U');
        
        % Subsample
        U=U(:, find(vt(:, big_T)), :);
        
        name_file_U_temp = [param.folder_file_U_temp 'dsamp_' num2str(n_subsampl_decor) '_' ...
            num2str(big_T) '_U_temp'];
        param_ref.name_file_U_temp = [param_ref.name_file_U_temp ...
            {name_file_U_temp}];
        % Save previous file with subsampled velocity
        save(name_file_U_temp, 'U');
    end
    param.name_file_U_temp = param_ref.name_file_U_temp;
else
    %% Load
    load(name_file_U_centered, 'U');
    
    len_blocks = param.data_in_blocks.len_blocks; % length of block before subsampling
    for t = 1 : param.N_tot % loop on time
        if t_local == len_blocks + 1 % A new file needs to be loaded
            
%             name_file_U_temp=[param.folder_file_U_temp param.type_data ...
%                             num2str(big_T) '_U_temp'];
            name_file_U_temp = [param.folder_file_U_temp 'dsamp_' num2str(n_subsampl_decor) '_' ...
                num2str(big_T) '_U_temp'];
            param_ref.name_file_U_temp = [param_ref.name_file_U_temp ...
                {name_file_U_temp}];
            
            % Save previous file with subsampled velocity
%             save(name_file_U_temp,'U');
            save(name_file_U_temp, 'U', '-v7.3');

            % initialization of the index of the snapshot before 
            % subsampling in the file
            t_local = 1;
            % initialization of the index of the snapshot after
            % subsampling in the file
            t_local_sub = 1;
            % Incrementation of the file index
            big_T = big_T + 1;
            
%             % Name of the new file
%             name_file_U_temp=param.name_file_U_temp{big_T};
%             % Load new file
%             load(name_file_U_temp);
            
            name_file_U_centered = [param.folder_data param.type_data ...
                num2str(big_T) '_U_centered'];
            % Load new file
            load(name_file_U_centered, 'U');
            
        end
        if mod(t, n_subsampl_decor)==0 %CRITICAL PART
            % This snapshot is kept by the subsampling
            t_local_sub = t_local_sub + 1;
        else % This snapshot is not kept by the subsampling
            U(:, t_local_sub, :) = [];
        end
        % Incrementation of the index of the snapshot in the file
        t_local = t_local + 1;
    end
    %% Save
    if param.data_in_blocks.bool
%         name_file_U_temp=[param.folder_file_U_temp param.type_data ...
%             num2str(big_T) '_U_temp'];
        name_file_U_temp = [param.folder_file_U_temp 'dsamp_' num2str(n_subsampl_decor) '_' ...
            num2str(big_T) '_U_temp'];
        param_ref.name_file_U_temp = [param_ref.name_file_U_temp ...
            {name_file_U_temp}];
        param.name_file_U_temp = param_ref.name_file_U_temp;
    else
%         name_file_U_temp=[param.folder_file_U_temp param.type_data ...
%              '_U_temp'];
        name_file_U_temp = [param.folder_file_U_temp 'dsamp_' num2str(n_subsampl_decor) '_' ...
             '_U_temp'];
        param.name_file_U_temp=name_file_U_temp;  
    end
    save(name_file_U_temp, 'U', '-v7.3');
    clear U;
end
