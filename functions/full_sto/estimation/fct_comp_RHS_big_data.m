function [R1,R2,R3] = fct_comp_RHS_big_data(param,bt,d_bt)

dt = param.dt;
N_tot = param.N_tot;
m = param.nb_modes;
nu = param.viscosity;
T = N_tot*dt;
M = param.M;
dX = param.dX;
MX = param.MX;
d = param.d;

% The last time step is not used
N_tot = N_tot - 1;
T = T -dt;

R1 = zeros(m,m);
R2 = zeros(m,m,m);
R3 = zeros(m,m,m,m);

%% load U

for i=1:m
    
    % Initialization
    t_local=1; % index of the snapshot in a file
    if param.data_in_blocks.bool % if data are saved in several files
        big_T = 1; % index of the file
        name_file_U_temp=param.name_file_U_temp{big_T}; % Name of the next file
    else
        name_file_U_temp=param.name_file_U_temp; % Name of the next file
    end
    
    load(name_file_U_temp);
    
    %compute the sum(dbt*U)
    del = zeros(M,1,d);
    %compute the sum(b_p*dbt_i*dX_res)
    del_pi = zeros(M,m,1,d);% (M,m(p),m(i),d)
    
    for t=1:N_tot % loop on time
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
        for k =1:d
            %         for i = 1:m
            %%
            del(:,1,k) = del(:,1,k) + U(:,t_local,k)*d_bt(t,i);
            for p = 1:m
                del_pi(:,p,1,k) = del_pi(:,p,1,k) ...
                    + U(:,t_local,k)*d_bt(t,i)*bt(t,p);
            end
            %%
%             del(:,1,k) = del(:,1,k) + U(:,t_local,k)*d_bt(t_local,i);
%             for p = 1:m
%                 del_pi(:,p,1,k) = del_pi(:,p,1,k) ...
%                     + U(:,t_local,k)*d_bt(t_local,i)*bt(t_local,p);
            end
            %%
            %         end
        end
        % Incrementation of the index of the snapshot in the file
        t_local=t_local+1;
    end
    
    % load(param.name_file_U_temp,'U');
    % if param.replication_data
    %     U = [U U U U U U];
    % end
    
    % %compute the sum(dbt*U)
    % del = zeros(M,m,d);
    % for k =1:d
    %     for i = 1:m
    %         for t = 1:N_tot
    %             del(:,i,k) = del(:,i,k) + U(:,t,k)*d_bt(t,i);
    %         end
    %
    %     end
    % end
    %
    % %compute the sum(b_p*dbt_i*dX_res)
    % del_pi = zeros(M,m,m,d);% (M,m(p),m(i),d)
    % for p = 1:m
    %     for k =1:d
    %         for i = 1:m
    %             for t = 1:N_tot
    %                 del_pi(:,p,i,k) = del_pi(:,p,i,k) + U(:,t,k)*d_bt(t,i)*bt(t,p);
    %             end
    %         end
    %     end
    % end
    clear U
    
    %% load phi
    load(param.name_file_mode,'phi_m_U')
    phi = phi_m_U(:,1:end-1,:);
    phi_0 = phi_m_U(:,end,:);
    clear phi_m_U;
    
    
    %% Compute R1
    
    %% compute the gradient of phi_0
    phi_0 = permute(phi_0,[2 3 1]);%(1,d,M)
    phi_0 = reshape(phi_0,[1 d MX]);%(1,d,Mx,My,(Mz))
    dphi_0 = gradient_mat(phi_0,dX);
    clear phi_0;
    dphi_0 = permute(dphi_0,[ndims(dphi_0)+1 1 ndims(dphi_0) 3:ndims(dphi_0)-1 2]);%(1,1,d!,Mx,My,(Mz),d
    
    % for i = 1:m
    % compute laplacian of del_i
    del_i = del(:,1,:);
    del_i = permute(del_i,[2 3 1]);
    del_i = reshape(del_i,[1 d MX]);%(1,d,Mx,My,(Mz))
    Lap_del_i = laplacian_mat(del_i,dX);
    Lap_del_i = nu*Lap_del_i;
    Lap_del_i = permute(Lap_del_i,[1 ndims(Lap_del_i)+1 2:ndims(Lap_del_i)]);
    Lap_del_i = permute(Lap_del_i, [1 2 4:ndims(Lap_del_i) 3]);%(1,1,Mx,My,(Mz),d)
    
    % compute the advection term_i ::adv_i
    del_i = permute(del_i,[ndims(del_i)+1 1:ndims(del_i)]);%(1,1,d,Mx ,My,(Mz))
    adv_i = bsxfun(@times,dphi_0,del_i); % 1 x (1) x d x Mx x My (x Mz) d
    clear del_i;
    adv_i = sum(adv_i,3);
    adv_i = permute(adv_i,[1 2 4:ndims(adv_i) 3]);%(1 1 Mx My (Mz) d)
    
    % compute the sum -adv_i and Lap_del_i
    sum_i= Lap_del_i - adv_i;
    clear Lap_del_i;
    clear adv_i;
    
    % Projection on free divergence space to remove the unknown
    % pressure term
    if param.eq_proj_div_free == 2
        sum_i = reshape(sum_i,[prod(MX) 1 d]);
        if strcmp(param.type_data, 'turb2D_blocks_truncated')
            sum_i = sum_i - proj_div_propre(sum_i,MX,dX, true);
        else
            sum_i = sum_i - proj_div_propre(sum_i,MX,dX, false);
        end
        sum_i = reshape(sum_i,[1 1 MX d]);
    end
    
    % projection on phi_j
    for j = 1:m % it should count from not the term 0
        phi_j = phi(:,j,:);
        phi_j = permute(phi_j,[4 2 1 3]);%(1,1,M,d)
        phi_j = reshape(phi_j,[1 1 MX d]);%(1,1,Mx,My,(Mz),d)
        
        s_temp = sum_i.*phi_j; %(1,1,Mx,My,(Mz),d)
        clear phi_j;
        s_temp = sum(s_temp,ndims(s_temp));%(1,1,Mx,My,(Mz))
        
        % compute the integration of s_temp
        s_temp = integration_mat(s_temp,dX);
        R1(i,j) = s_temp;
        clear s_temp;
    end
    clear sum_i;
    % end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Compute R2
    
    del_pi1 = del_pi;
    
    % compute laplacian of del_pi
    for p = 1:m
        %         for i = 1:m
        del_p_i = del_pi1(:,p,1,:);
        del_p_i = permute(del_p_i,[3 4  1 2]);%(1,d,M)
        del_p_i = reshape(del_p_i,[1 d MX]);%(1,d,Mx,My,(Mz))
        Lap_del_pi = laplacian_mat(del_p_i,dX);
        Lap_del_pi = nu*Lap_del_pi;
        Lap_del_pi = permute(Lap_del_pi,[1 ndims(Lap_del_pi)+1 2:ndims(Lap_del_pi)]);
        Lap_del_pi = permute(Lap_del_pi, [1 2 4:ndims(Lap_del_pi) 3]);%(1,1,Mx,My,(Mz),d)
        
        % compute the advection term_i ::adv_i
        del_p_i = permute(del_p_i,[ndims(del_p_i)+1 1:ndims(del_p_i)]);%(1,1,d,Mx ,My,(Mz))
        adv_pi = bsxfun(@times,dphi_0,del_p_i); % 1 x (1) x d x Mx x My (x Mz) d
        clear del_p_i;
        adv_pi = sum(adv_pi,3);
        adv_pi = permute(adv_pi,[1 2 4:ndims(adv_pi) 3]);%(1 1 Mx My (Mz) d)
        
        % compute the sum -adv_i and Lap_del_i
        sum_pi= Lap_del_pi - adv_pi;
        clear Lap_del_pi;
        clear adv_pi;
        
        % Projection on free divergence space to remove the unknown
        % pressure term
        if param.eq_proj_div_free == 2
            sum_pi = reshape(sum_pi,[prod(MX) 1 d]);
            if strcmp(param.type_data, 'turb2D_blocks_truncated')
                sum_pi = sum_pi - proj_div_propre(sum_pi,MX,dX, true);
            else
                sum_pi = sum_pi - proj_div_propre(sum_pi,MX,dX, false);
            end
            sum_pi = reshape(sum_pi,[1 1 MX d]);
        end
        
        % projection on phi_j
        for j = 1:m % it should count from not the term 0
            phi_j = phi(:,j,:);
            phi_j = permute(phi_j,[4 2 1 3]);%(1,1,M,d)
            phi_j = reshape(phi_j,[1 1 MX d]);%(1,1,Mx,My,(Mz),d)
            
            s_temp = sum_pi.*phi_j; %(1,1,Mx,My,(Mz),d)
            clear phi_j;
            s_temp = sum(s_temp,ndims(s_temp));%(1,1,Mx,My,(Mz))
            
            % compute the integration of s_temp
            s_temp = integration_mat(s_temp,dX);
            R2(p,i,j) = s_temp;
            
            
            clear s_temp;
        end
        clear sum_pi;
        %         end
    end
    clear del_pi1;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Compute R3
    
    %keyboard;
    
    for p = 1:m
        %         for i = 1:m
        del_p_i = del_pi(:,p,1,:);
        del_p_i = permute(del_p_i,[3 4  1 2]);%(1,d,M)
        del_p_i = reshape(del_p_i,[1 d MX]);%(1,d,Mx,My,(Mz))
        del_p_i = permute(del_p_i,[ndims(del_p_i)+1 1:ndims(del_p_i)]);%(1,1,d,Mx ,My,(Mz))
        
        % compute the gradient of phi_q
        for q = 1:m
            phi_q = phi(:,q,:);%(M,1,d)
            phi_q = permute(phi_q,[2 3 1]);%(1,d,M)
            phi_q = reshape(phi_q,[1 d MX]);%(1,d,Mx,My,(Mz))
            dphi_q = gradient_mat(phi_q,dX);
            clear phi_q;
            dphi_q = permute(dphi_q,[ndims(dphi_q)+1 1 ndims(dphi_q) 3:ndims(dphi_q)-1 2]);
            %(1,1,d!,Mx,My,(Mz),d
            
            % compute the advection term_piq ::adv_piq
            adv_piq = bsxfun(@times,dphi_q,del_p_i); % 1 x (1) x d x Mx x My (x Mz) d
            clear dphi_q;
            adv_piq = sum(adv_piq,3);
            adv_piq = permute(adv_piq,[1 2 4:ndims(adv_piq) 3]);%(1 1 Mx My (Mz) d)
            
            % Projection on free divergence space to remove the unknown
            % pressure term
            if param.eq_proj_div_free == 2
                adv_piq = reshape(adv_piq,[prod(MX) 1 d]);
                if strcmp(param.type_data, 'turb2D_blocks_truncated')
                    adv_piq = adv_piq - proj_div_propre(adv_piq,MX,dX, true);
                else
                    adv_piq = adv_piq - proj_div_propre(adv_piq,MX,dX, false);
                end
                adv_piq = reshape(adv_piq,[1 1 MX d]);
            end
            
            % projection on phi_j
            for j = 1:m % it should count from not the term 0
                phi_j = phi(:,j,:);
                phi_j = permute(phi_j,[4 2 1 3]);%(1,1,M,d)
                phi_j = reshape(phi_j,[1 1 MX d]);%(1,1,Mx,My,(Mz),d)
                
                s_temp = adv_piq.*phi_j; %(1,1,Mx,My,(Mz),d)
                clear phi_j;
                s_temp = sum(s_temp,ndims(s_temp));%(1,1,Mx,My,(Mz))
                
                % compute the integration of s_temp
                s_temp = integration_mat(s_temp,dX);
                R3(p,i,q,j) = - s_temp;
                
                clear s_temp;
            end
            clear adv_piq;
        end
        clear del_p_i;
        %         end
    end
    
end

clear del_pi;
R1 = R1*dt;
R2 = R2*dt;
R3 = R3*dt;
end