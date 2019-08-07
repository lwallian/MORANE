function param = gen_file_U_temp(param)
% Generate the teporary velocity file names
%

%% Initialization
param.folder_file_U_temp = fct_folder_temp(param);

% Subsampling rate
n_subsampl_decor = param.decor_by_subsampl.n_subsampl_decor;
t_local = 1; % index of the snapshot in a file before subsampling

if param.data_in_blocks.bool % if data are saved in several files
    big_T = 1; % index of the file
    param_ref.name_file_U_temp = {};
else
    param.data_in_blocks.len_blocks = param.N_tot;
end

if ~ (param.big_data || param.data_in_blocks.bool)
    param.name_file_U_temp=[param.folder_file_U_temp 'dsamp_' num2str(n_subsampl_decor) '_U_temp'];
elseif ~ param.big_data
    vt = zeros(1, param.N_tot);
    vt(1 : n_subsampl_decor : end) = true;
    vt = reshape(vt, [param.data_in_blocks.len_blocks ...
        param.data_in_blocks.nb_blocks]);
    param.data_in_blocks.mat_subsample = vt;
    for big_T = 1 : param.data_in_blocks.nb_blocks % loop on files        
        name_file_U_temp = [param.folder_file_U_temp 'dsamp_' num2str(n_subsampl_decor) '_' ...
            num2str(big_T) '_U_temp'];
        param_ref.name_file_U_temp = [param_ref.name_file_U_temp ...
            {name_file_U_temp}];
    end
    param.name_file_U_temp = param_ref.name_file_U_temp;
else
    len_blocks = param.data_in_blocks.len_blocks; % length of block before subsampling
    for t = 1 : param.N_tot % loop on time
        if t_local == len_blocks + 1 % A new file needs to be loaded
            name_file_U_temp = [param.folder_file_U_temp 'dsamp_' num2str(n_subsampl_decor) '_' ...
                            num2str(big_T) '_U_temp'];
            param_ref.name_file_U_temp = [param_ref.name_file_U_temp ...
                {name_file_U_temp}];

            % initialization of the index of the snapshot before 
            % subsampling in the file
            t_local = 1;
            % Incrementation of the file index
            big_T = big_T + 1;
        end
        % Incrementation of the index of the snapshot in the file
        t_local = t_local + 1;
    end
    %% Save
    if param.data_in_blocks.bool
        name_file_U_temp = [param.folder_file_U_temp 'dsamp_' num2str(n_subsampl_decor) '_' ...
            num2str(big_T) '_U_temp'];
        param_ref.name_file_U_temp = [param_ref.name_file_U_temp ...
            {name_file_U_temp}];
        param.name_file_U_temp = param_ref.name_file_U_temp;
    else
        name_file_U_temp = [param.folder_file_U_temp 'dsamp_' num2str(n_subsampl_decor) '_' ...
             '_U_temp'];      
        param.name_file_U_temp = name_file_U_temp;  
    end
end
