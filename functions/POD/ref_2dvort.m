function param = ref_2dvort(param,reconstruction)
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
if param.data_assimilation ==2 && (1/param.viscosity == 300)
    param.folder_file_U_fake_PIV = ...
        [ param.folder_data_PIV '/wake_Re' num2str(1/param.viscosity) ...
        '_fake/'];
    %     param.folder_file_U_fake_PIV = ...
    %         [ param.folder_file_U_temp(1:end-1) '_fake_PIV/'];
    if param.param_obs.no_noise
        param.folder_file_U_fake_PIV = [ param.folder_file_U_fake_PIV(1:end-1) ...
            '_noNoise/'];
    end
elseif (1/param.viscosity ~= 100)
    error(['Use one of the above cases : either refernce in the' ...
        '"DNS space" or in the "PIV space"']);
end


% Initialization
param = fct_name_reconstruction_2dvort(...
    param, nan, reconstruction,'ref');
% param.param_obs = nan;

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
first_big_T = big_T+1;
index_time = 0;
param.d = double(param.d);


if param.data_assimilation == 2 && (1/param.viscosity == 300)
    name_file_U_fake_PIV=[param.folder_file_U_fake_PIV 'strat' ...
        num2str(big_T+1) '_U_temp'];
    param.name_file_U_fake_PIV{1} = ...
        [ name_file_U_fake_PIV '_PIV'];
    load(param.name_file_U_fake_PIV{1},...
        'mask','x_PIV_after_crop','y_PIV_after_crop','MX_PIV',...
        'interval_time_local','dt');
    param.MX = MX_PIV;
    param.d = 2;
    param.dt = dt;
    n_subsampl = 1;
%     n_subsampl = ceil( dt/param.dt);
end

if ~param.DA.bool
    param.N_test=N_tot-1;
end
for t=1:n_subsampl*param.N_test % loop for all time
    if param.data_in_blocks.bool && ...
            ((t_local == len_blocks + 1) || t==1) % A new file needs to be loaded
        % initialization of the index of the snapshot in the file
        t_local=1;
        t_local_subsampl=1;
%         index_time = 0;
        v_index_time = [];
        % Incrementation of the file index
        big_T = big_T+1
        
        % Initialize U
        omega= [];
        
        % Load new file
        if param.data_assimilation == 2 && (1/param.viscosity == 300)
            name_file_U_fake_PIV=[param.folder_file_U_fake_PIV 'strat' ...
                num2str(big_T) '_U_temp'];
            param.name_file_U_fake_PIV{big_T - param.data_in_blocks.nb_blocks} = ...
                [ name_file_U_fake_PIV '_PIV'];
            load(param.name_file_U_fake_PIV{big_T - param.data_in_blocks.nb_blocks},...
                'U')
        else
            %         % Name of the new file
            param_temp.type_data=[param.data_in_blocks.type_whole_data num2str(big_T)];
            
            U=read_data(param_temp.type_data,param.folder_data, ...
                param.data_in_blocks.type_whole_data,param.modified_Re);
        end
        
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
            omega_U_temp = dU_temp(:,2,:,:,1) - dU_temp(:,1,:,:,2);
        elseif param.d == 3
            if param.data_assimilation == 2
                z_ref = 30;
            else
                z_ref = floor(param.MX(3)/2);
            end
            omega_U_temp = dU_temp(:,2,:,:,z_ref,1) - dU_temp(:,1,:,:,z_ref,2);
        else
            error('The dimension should be 2 or 3');
        end
        %
        %         if param.d == 2
        %             Omega_U_temp = dU_temp - permute( dU_temp , [1 5 3 4 2]);
        %         elseif param.d == 3
        %             Omega_U_temp = dU_temp - permute( dU_temp , [1 6 3 4 5 2]);
        %         else
        %             error('The dimension should be 2 or 3');
        %         end
        omega_U_temp = (1/2) * omega_U_temp;
        omega_U_temp = reshape(omega_U_temp, [1 1 prod(param.MX(1:2)) 1]);
        omega_U_temp = permute(omega_U_temp, [3 1 2 4]);
        
        omega = cat(2, omega, omega_U_temp );
        
        %%
        
        % Incrementation of the subsampled time
        t_subsample=t_subsample+1;
        t_local_subsampl=t_local_subsampl+1;
    end
    
    % Incrementation of the index of the snapshot in the file
    t_local=t_local+1;
    
    if param.data_in_blocks.bool && ...
            ((t_local == len_blocks + 1) )
        clear U
        %         Q = (1/2) * ...
        %             ( sum(sum(omega.^2,3),4) - sum(sum(S.^2,3),4) );
        omega = reshape(omega,[param.MX(1:2) size(omega,2)]);
        
        %% Plots
        [cmap,cax] = ...
            plot_iso_2dvort(param,'ref', nan, reconstruction, ...
            big_T,first_big_T, omega, v_index_time);
        if big_T == first_big_T
            param.plot.omega.cax = cax;
            param.plot.omega.cmap = cmap;
        end
        
        % A new file needs to be saved
        % Save
        name_file_temp =[ param.name_file_Reconstruction_omega ...
            num2str(big_T) '.mat'];
        param_from_file = param;
%         save(name_file_temp,'param_from_file','omega','-v7.3')
        
        clear omega
        
        %         if big_T > first_big_T
        %             warning('Returned forced');
        %            return;
        %         end
    end
end

if ~ ( param.data_in_blocks.bool && ...
        ((t_local == len_blocks + 1) ) )
    %% Part of the last file
    % (even if it was not read completely)
    
    clear U
    omega = reshape(omega,[param.MX(1:2) size(omega,2)]);
    
    %% Plots
%     index_time = (big_T-1)*len_blocks ;
%     index_time = (big_T-1)*len_blocks + t_local;
    [cmap,cax] = ...
        plot_iso_2dvort(param,'ref', nan, reconstruction, ...
        big_T,first_big_T, omega, v_index_time);
    if big_T == first_big_T
        param.plot.omega.cax = cax;
        param.plot.omega.cmap = cmap;
    end
    
    % A new file needs to be saved
    % Save
    name_file_temp =[ param.name_file_Reconstruction_omega ...
        num2str(big_T) '.mat'];
    param_from_file = param;
%     save(name_file_temp,'param_from_file','omega','-v7.3')
    
%     % omega_ref=omega;
%     % save('ref_omega.mat','omega_ref');
    
    clear omega
end
%%
