function [param,theta_dBt,alpha_dBt] = fct_estimator_U_res_all(param_ref,bt)

param = param_ref;
clear param_ref;
M = param.M;
d = param.d;
N_tot = param.N_tot;
dt = param.dt;
m = param.nb_modes;
MX = param.MX;
dX = param.dX;
nu = param.viscosity;
%keyboard;
%% compute dX_res
% load phi
load(param.name_file_mode,'phi_m_U');
% load U'
load(param.name_file_U_temp,'U');

% Define W = sum(bt,phi)
phi = phi_m_U(:,1:end-1,:);
phi_0 = phi_m_U(:,end,:);
clear phi_m_U;

% Define dX_res
if param.replication_data
    U = [U U U U U U];
    N_tot = 6*N_tot;
    param.N_tot = N_tot;
end
dX_res = U*dt;
clear U_res;

%% compute theta_dBt
theta_dBt = zeros(N_tot,m);
% Define laplacian dX_res_t
for t = 1:N_tot
    dX_temp = dX_res(:,t,:);
    dX_temp = permute(dX_temp,[2 3 1]);
    dX_temp = reshape(dX_temp,[1 d MX]);%(1,d,Mx,My,(Mz))
    %keyboard;
    % compute laplacian dX_temp
    Lap_dX = laplacian_mat(dX_temp,dX);
    Lap_dX = nu*Lap_dX;
    clear dX_temp;
    % Lap_dX = Lap_dX*nu;
    Lap_dX = permute(Lap_dX ,[1 ndims(Lap_dX)+1 2:ndims(Lap_dX)]);% (1) x 1 x d x Mx x My (x Mz)
    Lap_dX = permute(Lap_dX,[1 2 4:ndims(Lap_dX) 3]);%(1,1,Mx,My,(Mz),d)
    
    % projection on phi_i
    for i = 1:m
        phi_i = phi(:,i,:);
        phi_i = permute(phi_i,[4 2 1 3]);%(1,1,M,d)
        phi_i = reshape(phi_i,[1 1 MX d]);%(1,1,Mx,My,(Mz),d)
        
        % compute the scalar product or advertion
        ll_temp = Lap_dX.*phi_i;%(1,1,Mx,My,(Mz),d)
        clear phi_i;
        ll_temp = sum(ll_temp,ndims(ll_temp));%(1,1,Mx,My,(Mz))
        
        % compute the integration
        ll_temp = integration_mat(ll_temp,dX);
        theta_dBt(t,i) = ll_temp;
        clear ll_temp;
    end
    clear Lap_dX;
end

%     warning('trick');
%     theta_dBt=zeros(size(theta_dBt));

% % save theta_dBt
%     param.name_file_estimator_theta_dBt = [param.folder_data param.type_data '_theta_dBt'];
%     save(param.name_file_estimator_theta_dBt,'theta_dBt');
% % keyboard;
%% compute alpha_dBt::alpha_ji*dBt matrix (m+1,m,N_tot)=((m+1)^2,N_tot)
mode = m +1;
alpha_dBt = zeros(mode,m,N_tot);

for j = 1:mode
    if j == 1
        phi_j = phi_0;
        clear phi_0;
    else
        phi_j = phi(:,j-1,:);%(M,1,d)
    end
    
    phi_j = permute(phi_j,[2 3 1]);%(1,d,M)
    phi_j = reshape(phi_j,[1 d MX]);%(1,d,Mx,My,(Mz))
    
    % compute the gradient of each phi ((1),d!,Mx,My,(Mz),d)
    dphi = gradient_mat(phi_j,dX);
    clear phi_j;
    dphi = permute(dphi,[ndims(dphi)+1 1 ndims(dphi) 3:ndims(dphi)-1 2]);%(1,1,d!,Mx,My,(Mz),d
    %     dphi = permute(dphi,[ndims(dphi)+1 1:ndims(dphi)]);%(1,1,d!,Mx,My,(Mz),d
    
    % compute the advection term_t
    for t = 1:N_tot
        dX_temp = dX_res(:,t,:);
        dX_temp = permute(dX_temp,[4 2 3 1]);%(1,(1),d,M)
        dX_temp = reshape(dX_temp,[1 1 d MX]);%(1,1,d,Mx ,My,(Mz))
        ll_temp = bsxfun(@times,dphi,dX_temp); % 1 x (1) x d x Mx x My (x Mz) d
        ll_temp = sum(ll_temp,3);
        ll_temp = permute(ll_temp,[1 2 4:ndims(ll_temp) 3]);%(1 1 Mx My (Mz) d)
        clear dX_temp;
        %ll_temp = permute(sum(ll_temp,3),[1 2 4:ndims(ll_temp) 3]); % (1) x (1) x Mx x My (x Mz) x d
        for i = 1:m % it should count from not the term 0
            phi_i = phi(:,i,:);
            phi_i = permute(phi_i,[4 2 1 3]);%(1,1,M,d)
            phi_i = reshape(phi_i,[1 1 MX d]);%(1,1,Mx,My,(Mz),d)
            
            % compute the scalar product between phi_i and ll_temp projection on phi_i
            s_temp = ll_temp.*phi_i; %(1,1,Mx,My,(Mz),d)
            clear phi_i;
            s_temp = sum(s_temp,ndims(s_temp));%(1,1,Mx,My,(Mz))
            
            % compute the integration of s_temp
            s_temp = integration_mat(s_temp,dX);
            alpha_dBt(j,i,t) = - s_temp;
            clear s_temp;
            
            
        end
    end
end

%     warning('trick');
%     alpha_dBt(1,:,:)=0;
% %     alpha_dBt(2:end,:,:)=0;

% % save alpha_dBt
%     param.name_file_estimator_alpha_dBt = [param.folder_data param.type_data '_alpha_dBt'];
%     save(param.name_file_estimator_alpha_dBt,'alpha_dBt');
% 
% %keyboard;


end