function param = reconstruction_2dvort(param,bt,...
    name_simu,modal_dt,reconstruction)
% Reconstruct velocity field and save it
%

%% Load
param.name_file_omega_mode = [ param.folder_data ...
    '2dvort_mode_' param.type_data '_' num2str(param.nb_modes) '_modes.mat'];
load(param.name_file_omega_mode,'param_from_file',...
    'omega_phi_m_U');
% % Remove the time average value m_U
% mU = phi_m_U(:,param.nb_modes+1,:);
% phi_m_U(:,param.nb_modes+1,:)=[];
% phi=phi_m_U; clear phi_m_U
bt = cat( 2, bt, ones([size(bt,1),1]));

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
%     param.type_data = [ param.type_data(1:(end-10)) '_test_basis' ];
    param.data_in_blocks.type_whole_data = param.type_data;
%     % Load new file
%     U=read_data(param.type_data,param.folder_data, ...
%         param.data_in_blocks.type_whole_data,param.modified_Re);
%     U = bsxfun(@minus, U , m_U);
%     siz = size(U);
%     N_tot= floor(siz(end-1)/n_subsampl);
end


% Initialization
param = fct_name_reconstruction_2dvort(...
    param,modal_dt,reconstruction,name_simu);
t_local=1; % index of the snapshot in a file
%     t_local=0; % index of the snapshot in a file
t_subsample=1;
% %     if isfield(param,data_in_blocks.bool) && ...
% %             param.data_in_blocks.bool
% %         big_T = param.data_in_blocks.nb_blocks; % index of the file
% %     else
% %         big_T = [];
% %     end
% truncated_error2 = nan([1 N_tot]);
% bt = nan([N_tot 1 1 param.nb_modes]);
% U = zeros([param.M param_temp.data_in_blocks.len_blocks param.d]); % ???
big_T = param.data_in_blocks.nb_blocks; % index of the file
first_big_T = big_T + 1;

for t=1:n_subsampl*param.N_test % loop for all time
% for t=1:n_subsampl*N_tot % loop for all time
    if param.data_in_blocks.bool && ...
            ((t_local == len_blocks + 1) || t==1) % A new file needs to be loaded
        % initialization of the index of the snapshot in the file
        t_local=1;
        t_local_subsampl=1;
        % Incrementation of the file index
        big_T = big_T+1
%         % Name of the new file
%         param.type_data=[param.data_in_blocks.type_whole_data num2str(big_T)];
        
        % Initialization
        omega= [];
%         S= [];
%         Q = [];

%         U = zeros([param.M ...
%             param_temp.data_in_blocks.len_blocks param.d]);

%         % Load new file
%         U=read_data(param.type_data,param.folder_data, ...
%             param.data_in_blocks.type_whole_data,param.modified_Re);
%         
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
        % Add a snapshots
        omega = cat(2, omega, zeros([prod(param.MX(1:2)) 1 ]) );
%         S = cat(2, S, zeros([param.M 1 param.d param.d]) );
% %         Q = cat(2, S, zeros([param.M ]) );
        for p=1:(param.nb_modes+1)
            % Remove p-th resolved mode at time t
            omega(:,t_local_subsampl) = omega(:,t_local_subsampl) ...
                + omega_phi_m_U(:,p)*bt(t_subsample,p)';
%             S(:,t_local_subsampl,:,:) = S(:,t_local_subsampl,:,:) ...
%                 + S_phi_m_U(:,p,:,:)*bt(t_subsample,p)';
% %             for k=1:param.d
% %                 U(:,t_local_subsampl,k) = U(:,t_local_subsampl,k) ...
% %                     + phi_m_U(:,p,k)*bt(t_subsample,p)';
% %                 %                 U(:,:,k) = U(:,:,k) - phi_m_U(:,p,k)*bt(t,p)'
% %             end
        end
        
        
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
%         Q = (1/2) * ...
%             ( sum(sum(omega.^2,3),4) - sum(sum(S.^2,3),4) );
        omega = reshape(omega,[param.MX(1:2) size(omega,2)]);
        
        % A new file needs to be saved
        % Save
        name_file_temp =[ param.name_file_Reconstruction_omega ...
            num2str(big_T) '.mat'];
        param_from_file = param;
        save(name_file_temp,'param_from_file','omega','-v7.3')
        
        %% Plots
        plot_iso_2dvort(param,name_simu, modal_dt,reconstruction,...
            big_T,first_big_T, omega);
        clear omega
        
%         if big_T > first_big_T
%             warning('Returned forced');
%            return;
%         end
    end
end


%% Part of the last file
% (even if it was not read completely)
omega = reshape(omega,[param.MX(1:2) size(omega,2)]);

% A new file needs to be saved
% Save
name_file_temp =[ param.name_file_Reconstruction_omega ...
    num2str(big_T) '.mat'];
param_from_file = param;
save(name_file_temp,'param_from_file','omega','-v7.3')

%% Plots

% load('ref_omega.mat','omega_ref');
% omega = omega-omega_ref;

plot_iso_2dvort(param,name_simu, modal_dt,reconstruction,...
    big_T,first_big_T, omega);
clear omega


