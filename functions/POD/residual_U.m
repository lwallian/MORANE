function param = residual_U(param,bt)
% Keep only the residual velocity and save it
%

%% Load
load(param.name_file_mode)
% Remove the time average value m_U
phi_m_U(:,param.nb_modes+1,:)=[];
phi=phi_m_U; clear phi_m_U


%% Initialization
t_local=1; % index of the snapshot in a file
if param.data_in_blocks.bool % if data are saved in several files
    big_T = 1; % index of the file
    name_file_U_temp=param.name_file_U_temp{big_T}; % Name of the next file
else
    name_file_U_temp=param.name_file_U_temp; % Name of the next file
end


%% Remove resolved modes
if param.big_data
    %% Load
    load(name_file_U_temp);
    for t=1:param.N_tot % loop on time
        if t_local > size(U,2) % A new file needs to be loaded
            % Save previous file with residual velocity
            save(name_file_U_temp,'U','-v7.3');
            % initialization of the index of the snapshot in the file
            t_local=1;
            % Incrementation of the file index
            big_T = big_T+1;
            % Name of the new file
            name_file_U_temp=param.name_file_U_temp{big_T};
            % Load new file
            load(name_file_U_temp);
        end
        for p=1:param.nb_modes
            % Remove p-th resolved mode at time t
            for k=1:param.d
                U(:,t_local,k) = U(:,t_local,k) - phi(:,p,k)*bt(t,p)';
            end
        end
        % Incrementation of the index of the snapshot in the file
        t_local=t_local+1;
    end
    %% Save
    save(name_file_U_temp,'U','-v7.3');
    
elseif param.data_in_blocks.bool
    vt = param.data_in_blocks.mat_subsample;
    vt_block = sum(vt,1); % N_t of each block
    vt_block = cumsum(vt_block,2); 
    vt_block = [ 0 vt_block ];
    %     vt_block = [ 0 vt_block(1:end-1) ];
    
    for big_T=1:param.data_in_blocks.nb_blocks % loop on files
        % Name of the new file
        name_file_U_temp=param.name_file_U_temp{big_T};
        % Load new file
        load(name_file_U_temp);
        
        local_time_idx = (vt_block(big_T)+1):(vt_block(big_T+1));
        param.data_in_blocks.local_time_idx{big_T} = local_time_idx;
        
        for k=1:param.d
            U(:,:,k) = U(:,:,k) ...
                - phi(:,:,k)*bt(local_time_idx,:)';
        end
        
        % Save previous file with subsampled velocity
        save(name_file_U_temp,'U');
    end
    
else
    %% Load
    load(name_file_U_temp);
    for k=1:param.d
        U(:,:,k) = U(:,:,k) ...
            - phi(:,:,k)*bt';
    end
    %% Save
    save(name_file_U_temp,'U');
    
end
clear phi bt
