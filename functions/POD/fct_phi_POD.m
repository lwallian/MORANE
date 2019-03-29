function [param,phi] =fct_phi_POD(param,bt)
% Computation of spatial modes phi
%

% Initialization
phi = zeros(param.M,param.nb_modes,param.d);
if param.data_in_blocks.bool
    % index of the file
    big_T = 1;
%     param_ref.name_file_U_temp={};
%     % name of the first file
%     %     name_file_U_temp=param.name_file_U_centered{big_T};
%     % %     name_file_U_temp=param.name_file_U_temp{big_T};
%     name_file_U_temp=[param.folder_data param.type_data ...
%         num2str(big_T) '_U_temp'];

    % number of files used to save data
    nb_blocks=param.data_in_blocks.nb_blocks;
    % number of snapshots by file
    len_blocks=param.data_in_blocks.len_blocks;
    
    name_file_U_centered=[param.folder_data param.type_data ...
        num2str(big_T) '_U_centered'];
%     warning('temp -> centered');
    
else
    % name of the file
%     name_file_U_temp=param.name_file_U_centered;
% %     name_file_U_temp=param.name_file_U_temp;
    name_file_U_centered=[param.folder_data param.type_data ...
                             '_U_centered'];
                         
    param.data_in_blocks.len_blocks=param.N_tot;
end

load(name_file_U_centered);
if param.big_data 
% if param.big_data
    % index of the snapshot in the current file
    t_local=1;
    for t=1:param.N_tot
        if t_local == param.data_in_blocks.len_blocks + 1 
            % A new file needs to be loaded
            
            % initialization of the index of the snapshot in the file
            t_local=1;
            % Incrementation of the file index
            big_T = big_T+1;
            % Name of the new file
            name_file_U_centered=[param.folder_data param.type_data ...
                num2str(big_T) '_U_centered'];
            
            % Load new file
            load(name_file_U_centered);
        end
        % Projection of U on temporal modes bt
        for k=1:param.d
            phi(:,:,k) = phi(:,:,k) + U(:,t_local,k)*bt(t,:);
        end
        % Incrementation of the index of the snapshot in the file
        t_local=t_local+1;
    end
    
    clear U
%     param.name_file_U_temp=param_ref.name_file_U_temp;
    % Normalization
    for m=1:param.nb_modes
        phi(:,m,:) = 1/(param.N_tot*param.lambda(m)) * phi(:,m,:);
    end
elseif param.data_in_blocks.bool
% if param.big_data
    
    
    
    for big_T=1:nb_blocks % loop on files
            if big_T >= 2
                name_file_U_centered=[param.folder_data param.type_data ...
                    num2str(big_T) '_U_centered'];
                % Load new file
                load(name_file_U_centered);
            end
            
            % Set lines indexes of the block
            vt=(big_T-1)*len_blocks +1 : (big_T)*len_blocks;
            
            for k=1:param.d
                % Projection of U on temporal modes bt
                phi(:,:,k) = phi(:,:,k) + U(:,:,k)*bt(vt,:);
            end
    
    end
    
    for k=1:param.d
        % Normalization
        phi(:,:,k) = 1/param.N_tot * bsxfun(@times,param.lambda'.^(-1) ,  phi(:,:,k)); % param.M x nb_modes
    end
    
%     % index of the snapshot in the current file
%     t_local=1;
%     for t=1:param.N_tot
%         if t_local == param.data_in_blocks.len_blocks + 1 
%             % A new file needs to be loaded
%             
%             % initialization of the index of the snapshot in the file
%             t_local=1;
%             % Incrementation of the file index
%             big_T = big_T+1;
%             % Name of the new file
% %             name_file_U_temp=param.name_file_U_centered{big_T};
% % %             name_file_U_temp=param.name_file_U_temp{big_T};
% 
%             name_file_U_centered=[param.folder_data param.type_data ...
%                 num2str(big_T) '_U_centered'];
%             
%             % Load new file
%             load(name_file_U_centered);
%         end
%         % Projection of U on temporal modes bt
%         for k=1:param.d
%             phi(:,:,k) = phi(:,:,k) + U(:,t_local,k)*bt(t,:);
%         end
%         % Incrementation of the index of the snapshot in the file
%         t_local=t_local+1;
%     end
%     clear U
%     param.name_file_U_temp=param_ref.name_file_U_temp;
%
%     % Normalization
%     for m=1:param.nb_modes
%         phi(:,m,:) = 1/(param.N_tot*param.lambda(m)) * phi(:,m,:);
%     end

else
    for k=1:param.d
        % Projection of U on temporal modes bt
        phi(:,:,k) = U(:,:,k)*bt;
        % Normalization
        phi(:,:,k) = 1/param.N_tot * bsxfun(@times,param.lambda'.^(-1) ,  phi(:,:,k)); % param.M x nb_modes
    end
    
    %%
%     param.name_file_U_temp=[param.folder_data param.type_data '_U_temp'];
%     save(param.name_file_U_temp,'U','-v7.3');    
    %%
    
    clear U
end

%% Saving
% % phi_m_U = phi; % param.M x m x param.d
% % clear phi;
% load(param.name_file_mode,'m_U','phi_m_U');
% if exist('phi_m_U','var')
%     m_U=phi_m_U(:,param.nb_modes+1,:);
%     clear phi_m_U
% end
phi_m_U = phi; % param.M x m x param.d
clear phi;
if isfield(param.data_in_blocks,'type_whole_data') % if data are saved in several files
    type_data=[param.data_in_blocks.type_whole_data num2str(0)];
else
    type_data=[param.type_data num2str(0)];    
end
param.name_file_mU=[param.folder_data type_data '_U_centered'];
load(param.name_file_mU,'m_U');

phi_m_U(:,param.nb_modes+1,:)= squeeze(m_U); % param.M x (m+1) x param.d
clear m_U
save(param.name_file_mode,'phi_m_U');
% save(param.name_file_mode,'phi_m_U','-v7.3');

