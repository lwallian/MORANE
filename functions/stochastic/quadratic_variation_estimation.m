function [param,z]=quadratic_variation_estimation(param,bt)
% Variance tensor estimation
%

%% Get parameters
param = fct_name_file_diffusion_mode(param);

param_temp = param;
param_temp.a_time_dependant = true;
param_temp = fct_name_file_diffusion_mode(param_temp);

if (exist(param.name_file_diffusion_mode,'file')==2) || ...
        (exist(param_temp.name_file_diffusion_mode,'file')==2)
    if ~ (exist(param.name_file_diffusion_mode,'file')==2)
        load(param_temp.name_file_diffusion_mode,'z');
        z(:,1:param.nb_modes,:,:) = [];
        save(param.name_file_diffusion_mode,'z');
    end
else
    clear param_temp
    %%
    
    % name_file_mode=param.name_file_mode; % file name for the spatial modes
    big_data=param.big_data;
    if param.data_in_blocks.bool % if data are saved in several files
        size_of_a_block = param.d * prod(param.MX) * ...
            param.data_in_blocks.len_blocks ...
            / param.decor_by_subsampl.n_subsampl_decor
        memory_of_a_block = size_of_a_block*64/8
        if memory_of_a_block * param.d * param.nb_modes > 10e9 
%         if (param.decor_by_subsampl.n_subsampl_decor == 1) ...
%                 && ( param.nb_modes > 4 )
            warning('The data are too big. So the algorithm will be slower');
            big_data = true
        end
    end
    % model for the variance tensor a
    a_time_dependant=param.a_time_dependant;
    if a_time_dependant
        type_filter_a=param.type_filter_a;
    end
    M=param.M;
    N_tot=param.N_tot;
    d=param.d;
    dt=param.dt;
    lambda=param.lambda; % Energy of Chronos
    
    if isfield(param,'N_estim')
        param.N_estim=ceil(param.N_estim/param.decor_by_subsampl.n_subsampl_decor);
        nb_Strouhal=0.2;
        param.period_estim= floor(param.N_estim*param.dt*nb_Strouhal);
        N_tot=param.N_estim;
        bt=bt(1:N_tot,:);
        fprintf(['estimation of the variance tensor with only ' ...
            num2str(N_tot) ' snapshots']);
    end
    
    %% Initialization
    if a_time_dependant && strcmp(type_filter_a,'b_i')
        % a(x,t) = z_0(x) + \sum b_i(t) z_i(x)
        % number of modes z_j
        nb_modes_z = param.nb_modes+1;
        % Projection basis
        % Add the constant 1 to Chronos
        bt = [ bt'; ones(1,N_tot)];% nb_modes_z x N_tot
        % Energy of the projection basis
        lambda= [lambda; 1]; % nb_modes_z
    elseif ~a_time_dependant
        % a(x,t) = z_0(x)
        % number of modes z_j
        nb_modes_z =1;
        % Projection basis
        % Use only the constant 1
        bt = ones(1,N_tot);% nb_modes_z x N_tot
        % Energy of the projection basis
        lambda=1;
    else
        error('not coded or deprecated');
    end
    
    t_local=1; % index of the snapshot in a file
    if param.data_in_blocks.bool % if data are saved in several files
        big_T = 1; % index of the file
        name_file_U_temp=param.name_file_U_temp{big_T}; % Name of the next file
    else
        name_file_U_temp=param.name_file_U_temp; % Name of the next file
    end
    % Load
    load(name_file_U_temp);
    
    z=zeros(M,nb_modes_z,d,d);
    big_T_max = size(param.name_file_U_temp,2); %BETA PARAMETER
    %% Projection
    if big_data
        for t=1:N_tot
            if t_local > size(U,2) % A new file needs to be loaded
                % initialization of the index of the snapshot in the file
                t_local=1;
                % Incrementation of the file index
                big_T = big_T+1;
                if big_T > big_T_max %BETA IF CLAUSE
                    break
                end
                % Name of the new file
                name_file_U_temp=param.name_file_U_temp{big_T};
                % Load new file
                load(name_file_U_temp);
            end
            % Projection
            for k_mode=1:nb_modes_z
                for i=1:d
                    for j=1:d
                        z(:,k_mode,i,j)=z(:,k_mode,i,j) ...
                            + bt(k_mode,t)/lambda(k_mode) ...
                            * U(:,t_local,i) .* U(:,t_local,j);% M x 1 x d x d
                    end
                end
            end
            % Incrementation of the index of the snapshot in the file
            t_local=t_local+1;
        end
        clear U;
    elseif param.data_in_blocks.bool
        
        
        for big_T=1:param.data_in_blocks.nb_blocks % loop on files
            local_time_idx = param.data_in_blocks.local_time_idx{big_T};
            if ~isempty(local_time_idx) && ( ~isfield(param,'N_estim') ...
                                           || (local_time_idx(1) > N_tot) )
                % Name of the new file
                name_file_U_temp=param.name_file_U_temp{big_T};
                % Load new file
                load(name_file_U_temp);
                
                if local_time_idx(end) > N_tot
%                     local_time_idx_temp = local_time_idx;
                    local_time_idx = local_time_idx(1):N_tot;
                    U((length(local_time_idx)+1):end) = [];
                end
                % Compute U(x,t) U(x,t)'
                a = bsxfun(@times,U,permute(U,[1 2 4 3])); % M x N_local x d x d
                clear U
                % Projection
                for j=1:nb_modes_z
                    z_temp = sum(bsxfun(@times,a,bt(j,local_time_idx)/lambda(j)), 2);
                    % M x 1 x d x d
                    z(:,j,:,:) = z(:,j,:,:) + z_temp;% M x 1 x d x d
                    
%                     z_temp = bsxfun(@times,a,bt(j,local_time_idx)/lambda(j));
%                     % M x N x d x d
%                     z(:,j,:,:) = z(:,j,:,:) + sum(z_temp, 2);% M x 1 x d x d
%                     clear z_temp
                end
                clear a
            end
        end
        
        
    else
        % Compute U(x,t) U(x,t)'
        U=U(:,1:N_tot,:);
        a = bsxfun(@times,U,permute(U,[1 2 4 3])); % M x N x d x d
        clear U
        % Projection
        for j=1:nb_modes_z
            z_temp = bsxfun(@times,a,bt(j,:)/lambda(j)); % M x N x d x d
            z(:,j,:,:)=sum(z_temp, 2);% M x 1 x d x d
        end
        clear a
    end
    % Normalization and time step influence
    z=dt/N_tot*z;
    
    % To circumvent the effect of the threshold of the downsampling rate
    if strcmp(param.decor_by_subsampl.choice_n_subsample, 'corr_time')
        z = z * param.decor_by_subsampl.tau_corr / param.decor_by_subsampl.n_subsampl_decor;
    end
    
    %% Save
    if nargout < 2
        save(param.name_file_diffusion_mode,'z','-v7.3');
        %     save(name_file_mode,'z','-append');
        clear z
    end
end