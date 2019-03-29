function fct_m_U(param)
% Compute time average value on all snapshots (taking into account all files)
%

if isfield(param.data_in_blocks,'type_whole_data') % if data are saved in several files
    % Get some information on how the data are saved
    param_blocks=read_data_blocks(param.data_in_blocks.type_whole_data, ...
        param.folder_data);
    % number of files used to save data
    nb_blocks=param_blocks.data_in_blocks.nb_blocks;
    % number of snapshots by file
    len_blocks=param_blocks.data_in_blocks.len_blocks;
    % total number of snapshots
    N_tot=len_blocks*nb_blocks;
    
    % Initialization
    m_U = 0; % time average value
    
    if param.big_data
        t_local=0; % index of the snapshot in a file
        big_T = 0; % index of the file
        for t=1:N_tot % loop for all time
            if (t_local == len_blocks + 1) || t==1 % A new file needs to be loaded
                % initialization of the index of the snapshot in the file
                t_local=1;
                % Incrementation of the file index
                big_T = big_T+1;
                % Name of the new file
                param.type_data=[param.data_in_blocks.type_whole_data num2str(big_T)];
                % Load new file
                [U,param_dX]=read_data(param.type_data,param.folder_data, ...
                    param.data_in_blocks.type_whole_data);
                warning('The Reynolds number is not modified');
                %             [U,param_dX]=read_data(param.type_data,param.folder_data, ...
                %                 param.data_in_blocks.type_whole_data,param.modified_Re);
            end
            % Summing the current snapshot to the average value
            m_U = m_U + U(:,t_local,:);
            % Incrementation of the index of the snapshot in the file
            t_local=t_local+1;
        end
        clear U
    else
        for big_T=1:nb_blocks % loop for all time
            % Name of the new file
            param.type_data=[param.data_in_blocks.type_whole_data num2str(big_T)];
            % Load new file
            [U,param_dX]=read_data(param.type_data,param.folder_data, ...
                param.data_in_blocks.type_whole_data);
            % Summing the current snapshots to the average value
            m_U = m_U + sum(U,2);
            clear U
        end
    end
    m_U=1/N_tot*m_U;
else % if data are saved in only one file
    % Load data
    [U,param_dX]=read_data(param.type_data,param.folder_data);
    % Averaging
    m_U = mean(U,2);
    clear U
end

%% Save
if isfield(param.data_in_blocks,'type_whole_data') % if data are saved in several files
    type_data=[param.data_in_blocks.type_whole_data num2str(0)];
else
    type_data=[param.type_data num2str(0)];    
end
param.name_file_mU=[param.folder_data type_data '_U_centered'];
save(param.name_file_mU,'m_U');
% save(param.name_file_mode,'m_U');

m_U(:,:,1)=m_U(:,:,1)-1;
nrj_mU = sum(m_U(:).^2)*prod(param_dX.dX);
param.file_nrj_mU=[ param.folder_results '_nrj_mU_' type_data '.mat'];
% param.file_nrj_mU=[ param.folder_results '_nrj_mU_' param.type_data '.mat'];
save(param.file_nrj_mU, 'nrj_mU');