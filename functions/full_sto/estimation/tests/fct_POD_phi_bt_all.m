function [param,bt] = fct_POD_phi_bt_all(param_ref)

%keyboard;
%     [~,param_temp] = read_data(param_ref.type_data,param_ref.folder_data);
%     param_temp=rmfield(param_temp,'type_data');
%     param_temp = rmfield(param_temp,'folder_data');
%     param = mergestruct(param_ref,param_temp);
%     clear param_temp;
%     clear param_ref;
    [param,c] = fct_mU_U_temp_matrix_c(param_ref);
% % load matrix c
%     param.name_file_matrix_c = [param.folder_U_t_test param.type_data '_c'];
%     load(param.name_file_matrix_c);

    param.d = length(param.dX);
    param.N_tot = size(c,1);
    c=c*prod(param.dX)/param.N_tot;
    nb_modes = param.nb_modes; 
    [W,S]=eigs(c,nb_modes);
    lambda=diag(S);clear S
   % lambda = max(lambda,0)
    
    
%% Computation of the Chronos b(t)
    bt=sqrt(param.N_tot) * W * sqrt(diag(lambda)); % temporal modes % N x nb_modes
    clear W;
    clear c;

% Force the convention: bt(1,:) > 0
% It is then easier to compare results of several simulation
    idx = bt(1,:)< 0;
    if  any(idx)
        idx=find(idx);
        bt(:,idx) = - bt(:,idx);
    end

% Keep only the the first nb_modes values
    bt=bt(:,1:nb_modes);
    lambda=lambda(1:nb_modes);
    param.lambda = lambda;
    clear lambda;

% % save bt
%     param.name_file_bt = [param.folder_data param.type_data '_bt'];
%     save(param.name_file_bt,'bt','-v7.3');
   
%% compute phi
%keyboard;
% load U'
    %param.name_file_U_t_test = [param.folder_U_t_test param.type_data '_U_test'];
    load(param.name_file_U_t_test);
    param.M = size(U,1);
    phi = zeros(param.M,param.nb_modes,param.d);
    for k=1:param.d
        % Projection of U on temporal modes bt
        phi(:,:,k) = U(:,:,k)*bt;
        % Normalization
        phi(:,:,k) = 1/param.N_tot * bsxfun(@times,param.lambda'.^(-1) ,phi(:,:,k)); % param.M x nb_modes
    end
    clear U

%keyboard;
% load m_U = phi_0
    %param.name_file_m_U = [param.folder_U_t_test param.type_data '_m_U'];
    load(param.name_file_m_U);
    phi(:,param.nb_modes+1,:) = squeeze(m_U);
    phi_mu = [phi(:,param.nb_modes+1,:),phi(:,1:param.nb_modes,:)];
    clear phi;
    clear m_U;
 
 % save phi
    param.name_file_phi = [param.folder_data param.type_data '_phi'];
    save(param.name_file_phi,'phi_mu','-v7.3');
    clear phi_mu;
    %clear bt; 
end