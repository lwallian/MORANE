function param = ref_Q(param,reconstruction)
% Reconstruct velocity field and save it
%

global stochastic_integration;
global estim_rmv_fv;
global choice_n_subsample;
global correlated_model;

%%
n_subsampl = param.decor_by_subsampl.n_subsampl_decor;

if isfield(param,'data_in_blocks') && ...
        isfield(param.data_in_blocks,'bool') && ...
        param.data_in_blocks.bool
    
    % if data are saved in several files
    len_blocks=param.data_in_blocks.len_blocks;
    %         param_temp = read_data_blocks(...
    %             [ param.type_whole_data(1:(end-10)) '_test_basis' ],...
    %             param.folder_data);
    param_temp = read_data_blocks(...
        [ param.type_data(1:(end-10)) '_test_basis' ],...
        param.folder_data);
    N_tot= param_temp.data_in_blocks.nb_blocks ...
        * param_temp.data_in_blocks.len_blocks;
%     clear param_temp
    N_tot= floor(N_tot/n_subsampl);
else
    %         param.data_in_blocks.bool = false;
    %         len_blocks=inf;
    %         param.data_in_blocks.nb_blocks = [];
    %         param.type_data = [ param.type_data(1:(end-10)) '_test_basis' ];
    %         param.data_in_blocks.type_whole_data = param.type_data;
    param.data_in_blocks.bool = false;
    len_blocks=inf;
    param.data_in_blocks.nb_blocks = [];
    param.N_tot = param.N_test + 1;
    param_temp.data_in_blocks.len_blocks = param.N_tot;
    param_temp.type_data = [ param.type_data(1:(end-10)) '_test_basis' ];
    param.data_in_blocks.type_whole_data = param.type_data;
%     % Load new file
%     U=read_data(param.type_data,param.folder_data, ...
%         param.data_in_blocks.type_whole_data,param.modified_Re);
%     U = bsxfun(@minus, U , m_U);
%     siz = size(U);
%     N_tot= floor(siz(end-1)/n_subsampl);
end


% Initialization
param = fct_name_reconstruction_Q(...
    param, nan, reconstruction,'ref');
% param = fct_name_ref_Q(...
%     param,reconstruction);
t_local=1; % index of the snapshot in a file
%     t_local=0; % index of the snapshot in a file
t_subsample=1;
v_threshold= nan;
% %     if isfield(param,data_in_blocks.bool) && ...
% %             param.data_in_blocks.bool
% %         big_T = param.data_in_blocks.nb_blocks; % index of the file
% %     else
% %         big_T = [];
% %     end
% truncated_error2 = nan([1 N_tot]);
% bt = nan([N_tot 1 1 param.nb_modes]);
% U = zeros([param.M param_temp.data_in_blocks.len_blocks param.d]); % ???
big_T = double(param.data_in_blocks.nb_blocks); % index of the file
param.MX = double(param.MX);
param.M = double(param.M);
param.d = double(param.d); 
first_big_T = big_T+1;
index_time = 0;

for t=1:n_subsampl*param.N_test % loop for all time
    if param.data_in_blocks.bool && ...
            ((t_local == len_blocks + 1) || t==1) % A new file needs to be loaded
        % initialization of the index of the snapshot in the file
        t_local=1;
        t_local_subsampl=1;
        v_index_time = [];
        % Incrementation of the file index
        big_T = big_T+1
%         % Name of the new file
        param_temp.type_data=[param.data_in_blocks.type_whole_data num2str(big_T)];
        
        % Initialize U
        Omega= [];
        S= [];
        
        % Load new file
        U=read_data(param_temp.type_data,param.folder_data, ...
            param.data_in_blocks.type_whole_data,param.modified_Re);
        
%         U = bsxfun(@minus, U , m_U);
%         
%         %             if t ==1 && ...
%         %                 ~ ( isfield(param,'data_in_blocks') && ...
%         %                     isfield(param.data_in_blocks,'bool') && ...
%         %                     param.data_in_blocks.bool )
%         %                 siz = size(U);
%         %                 N_tot= siz(end-1);
%         %                 truncated_error2 = nan([1 N_tot]);
%         %                 bt = nan([N_tot 1 1 param.nb_modes]);
%         %             end
    end
    
    if mod(t-1,n_subsampl)==0
        index_time = index_time + 1;
        v_index_time = [ v_index_time index_time ];
        
        % Add a snapshots        
        
        U_temp = permute(U(:,t_local,:),[2 3 1]); % 1 x d x M
        % 1 x d(number of velocity components) x M(space)
        U_temp = reshape(U_temp, [1 param.d param.MX]); % 1 x d x Mx x My (x Mz)
        
        % Gradient of the velocity snapshot % 1 x d x Mx x My (x Mz) x d
        dU_temp = gradient_mat(U_temp,param.dX);
        
        %%
%         Q=reshape(dU_temp(1,2,:,:,:,1),param.MX);
%         slice_Q = Q(:,:,floor(size(Q,3)/2),1);
%         x=param.dX(1)*(0:(param.MX(1)-1));
%         y=param.dX(2)*(0:(param.MX(2)-1));y=y-mean(y);
%         figure;imagesc(x,y,slice_Q');axis xy;axis equal;
%         keyboard;
        
        %%
        
        % velocity snapshot vorticity tensors
        if param.d == 2
            Omega_U_temp = dU_temp - permute( dU_temp , [1 5 3 4 2]);
        elseif param.d == 3
            Omega_U_temp = dU_temp - permute( dU_temp , [1 6 3 4 5 2]);
        else
            error('The dimension should be 2 or 3');
        end
        Omega_U_temp = (1/2) * Omega_U_temp;
        Omega_U_temp = reshape(Omega_U_temp, [1 param.d param.M param.d]);
        Omega_U_temp = permute(Omega_U_temp, [3 1 2 4]);
        
        Omega = cat(2, Omega, Omega_U_temp );
        
        %%
%         Q=reshape(dU_temp(1,1,:,:,:,1),param.MX);
%         slice_Q = Q(:,:,floor(size(Q,3)/2),1);
%         x=param.dX(1)*(0:(param.MX(1)-1));
%         y=param.dX(2)*(0:(param.MX(2)-1));y=y-mean(y);
%         figure;imagesc(x,y,slice_Q');axis xy;axis equal;
%         keyboard;
        
        %%
        
        % velocity snapshot rate-of-strain tensors
        if param.d == 2
            S_U_temp = dU_temp + permute( dU_temp , [1 5 3 4 2]);
        elseif param.d == 3
            S_U_temp = dU_temp + permute( dU_temp , [1 6 3 4 5 2]);
        else
            error('The dimension should be 2 or 3');
        end
        S_U_temp = (1/2) * S_U_temp;
%         S_U_temp = reshape(S_U_temp,...
%             [param.M 1 param.d param.d]);
        S_U_temp = reshape(S_U_temp, [1 param.d param.M param.d]);
        S_U_temp = permute(S_U_temp, [3 1 2 4]);
        
        S = cat(2, S, S_U_temp );
        
%         % Add a snapshots
%         Omega = cat(2, Omega, zeros([param.M 1 param.d param.d]) );
%         S = cat(2, S, zeros([param.M 1 param.d param.d]) );
% %         Q = cat(2, S, zeros([param.M ]) );
%         for p=1:(param.nb_modes+1)
%             % Remove p-th resolved mode at time t
%             Omega(:,t_local_subsampl,:,:) = Omega(:,t_local_subsampl,:,:) ...
%                 + Omega_phi_m_U(:,p,:,:)*bt(t_subsample,p)';
%             S(:,t_local_subsampl,:,:) = S(:,t_local_subsampl,:,:) ...
%                 + S_phi_m_U(:,p,:,:)*bt(t_subsample,p)';
% %             for k=1:param.d
% %                 U(:,t_local_subsampl,k) = U(:,t_local_subsampl,k) ...
% %                     + phi_m_U(:,p,k)*bt(t_subsample,p)';
% %                 %                 U(:,:,k) = U(:,:,k) - phi_m_U(:,p,k)*bt(t,p)'
% %             end
%         end
            
        
        
%         %         if mod(t,n_subsampl)==1
%         bt(t_subsample,1,1,:) = sum(sum(bsxfun(@times, U(:,t_local,:), phi_m_U),1),3) ...
%             * prod(param.dX);
%         
%         U(:,t_local,:) = U(:,t_local,:) ...
%             - sum( bsxfun(@times, phi_m_U, bt(t_subsample,1,1,:)) ,4);
%         
%         truncated_error2(1,t_subsample) = sum(sum(U(:,t_local,:).^2,1),3) ...
%             * prod(param.dX);
        
        % Incrementation of the subsampled time
        t_subsample=t_subsample+1;
        t_local_subsampl=t_local_subsampl+1;
    end
    
    % Incrementation of the index of the snapshot in the file
    t_local=t_local+1;
    
    
    if param.data_in_blocks.bool && ...
            ((t_local == len_blocks + 1) ) 
        clear U
        Q = (1/2) * ...
            ( sum(sum(Omega.^2,3),4) - sum(sum(S.^2,3),4) );
        Q = reshape(Q,[param.MX size(Q,2)]);
        
        %% Plots
        v_threshold = ...
            plot_isoQ(param,'ref', nan, reconstruction, ...
            big_T,first_big_T, Q, v_index_time);
        if big_T == first_big_T
            param.plot.Q.v_threshold = v_threshold;
        end
        
        % A new file needs to be saved
        % Save
%         name_file_temp =[ param.name_file_Reconstruction_Q ...
%             num2str(big_T) '.mat'];
%         param_from_file = param;
%         save(name_file_temp,'param_from_file','Q','-v7.3')
        
        clear Omega S Q
        
%         if big_T > first_big_T
%             warning('Returned forced');
%            return;
%         end
    end
end

%% Part of the last file
% (even if it was not read completely)
clear U
Q = (1/2) * ...
    ( sum(sum(Omega.^2,3),4) - sum(sum(S.^2,3),4) );
Q = reshape(Q,[param.MX size(Q,2)]);

%% Plots
v_threshold = ...
    plot_isoQ(param,'ref', nan, reconstruction, ...
    big_T,first_big_T, Q, v_index_time);
if big_T == first_big_T
    param.plot.Q.v_threshold = v_threshold;
end

% A new file needs to be saved
% Save
% name_file_temp =[ param.name_file_Reconstruction_Q ...
%     num2str(big_T) '.mat'];
% param_from_file = param;
% save(name_file_temp,'param_from_file','Q','-v7.3')

clear Omega S Q


