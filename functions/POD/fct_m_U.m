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
%                 warning('The Reynolds number is not modified');
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

%% Energy of the mean computation (for post-procesing plots)
switch type_data(1:(end-1))
    case {'incompact3d_wake_episode3_cut_truncated',...
            'incompact3D_noisy2D_40dt_subsampl',...
            'incompact3D_noisy2D_40dt_subsampl_truncated',...
            'inc3D_Re300_40dt_blocks', 'inc3D_Re300_40dt_blocks_truncated', 'inc3D_Re300_40dt_blocks_test_basis',...
            'inc3D_Re3900_blocks', 'inc3D_Re3900_blocks_truncated', 'inc3D_Re3900_blocks_test_basis',...
            'DNS300_inc3d_3D_2017_04_02_blocks', 'DNS300_inc3d_3D_2017_04_02_blocks_truncated',...
            'DNS300_inc3d_3D_2017_04_02_blocks_test_basis',...
            'test2D_blocks', 'test2D_blocks_truncated', 'test2D_blocks_test_basis',...
            'small_test_in_blocks', 'small_test_in_blocks_truncated',...
            'small_test_in_blocks_test_basis',...
            'DNS100_inc3d_2D_2018_11_16_blocks',...
            'DNS100_inc3d_2D_2018_11_16_blocks_truncated',...
            'DNS100_inc3d_2D_2018_11_16_blocks_test_basis',...
             'DNS100_OpenFOAM_2D_2020_blocks',...
             'DNS100_OpenFOAM_2D_2020_blocks_truncated',...
             'DNS100_OpenFOAM_2D_2020_blocks_test_basis',...
            'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks',...
            'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_truncated',...
            'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_test_basis',...
            'inc3D_HRLESlong_Re3900_blocks',...
            'inc3D_HRLESlong_Re3900_blocks_truncated',...
            'inc3D_HRLESlong_Re3900_blocks_test_basis'}
         m_U(:,:,1)=m_U(:,:,1)-1;
end

%
nrj_mU = sum(m_U(:).^2)*prod(param_dX.dX);
param.file_nrj_mU=[ param.folder_results '_nrj_mU_' type_data '.mat'];
% param.file_nrj_mU=[ param.folder_results '_nrj_mU_' param.type_data '.mat'];
mkdir(param.folder_results)
save(param.file_nrj_mU, 'nrj_mU');