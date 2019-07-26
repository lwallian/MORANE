function result = estimation_test_all(param,bt,bool_theta,bool_alpha)
% This function estimate the covariance of the additive and multiplicative
% noises, without assuming that the Chronos are orthogonal

M = param.M;
dX = param.dX;
MX = param.MX;
d = param.d;
m = param.nb_modes;
N_tot = param.N_tot;
dt = param.dt;
T = dt*N_tot;
%     param = param;
%     clear param;

% The last time step is not used
N_tot = N_tot - 1;
T = T -dt;

d_bt = bt(2:end,:)-bt(1:end-1,:);

%% test without considering on alpha_dBt
%keyboard;
if (bool_theta == true && bool_alpha == false)
    MM = 0;
    NN = 0;
    int_b = 0;
    theta_theta = zeros(m,m);
    
    % Compute the right hand-side
    for i = 1:m
        for j = 1:m
            R = 0;
            for t = 1:N_tot
                R = R + d_bt(t,i)*d_bt(t,j);
            end
            theta_theta(i,j) = R/T;
        end
    end
    result = theta_theta;
    clear theta_theta;
    %keyboard;
elseif (bool_theta == false && bool_alpha == true)
    %% Compute the RHS of equation
    % of course, we have the form M_?p*alpha_?i*alpha_qj =-RHS(p,i,j,q)
    
    % Define Matrix M
    MM = fct_comp_mat_M(param,bt);
    NN = 0;
    int_b = 0;
        
    % load U'
    load(param.name_file_U_temp,'U');
    
    if param.replication_data
        U = [U U U U U U];
    end
    
    % load phi
    load(param.name_file_mode,'phi_m_U');
    phi = phi_m_U(:,1:end-1,:);
    clear phi_mu;
    
    % compute the sum_t(U*d_bt*bt)
    del_pi = zeros(M,m,m,d);% (M,m(p),m(i),d)
    for p = 1:m
        del_p = zeros(M,m,d);
        for k =1:d
            for i = 1:m
                for t = 1:N_tot
                    del_p(:,i,k) = del_p(:,i,k) + U(:,t,k)*d_bt(t,i)*bt(t,p)*dt;
                end
            end
        end
        del_p = permute(del_p,[1 ndims(del_p)+1 2:ndims(del_p)]); % MX x 1 x m x d
        del_pi(:,p,:,:) = del_p;% MX x m x m x d
        clear del_p;
    end
    clear U
    
    %keyboard;
    RHS = zeros(m,m,m,m);
    
    for p = 1:m
        for i = 1:m
            del_p_i = del_pi(:,p,i,:);
            del_p_i = permute(del_p_i,[3 4 1 2]);%(1,d,M)
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
%                 dphi_q = permute(dphi_q,[ndims(dphi_q)+1 1:ndims(dphi_q)]);%(1,1,d!,Mx,My,(Mz),d
                
                % compute the advection term_piq ::adv_piq
                adv_piq = bsxfun(@times,dphi_q,del_p_i); % 1 x (1) x d x Mx x My (x Mz) d
                clear dphi_q;
                adv_piq = sum(adv_piq,3);
                adv_piq = permute(adv_piq,[1 2 4:ndims(adv_piq) 3]);%(1 1 Mx My (Mz) d)
                
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
                    RHS(p,i,q,j) = - s_temp;
                    %alpha_alpha(p,i,q,j) = - s_temp/T;
                    clear s_temp;
                end
                clear adv_piq;
            end
            clear del_p_i;
        end
    end
    clear del_pi;
    
    % compute the estimation alpha_alpha
    alpha_alpha = zeros(m,m,m,m);
    for j = 1:m
        for q = 1:m
            for i = 1:m
                R_ip = RHS(:,i,q,j);
%                 R_ip = MM'*R_ip;
%                 MM = MM*MM';
                alpha_temp = linsolve(MM,R_ip);
                alpha_alpha(:,i,q,j) = alpha_temp;
                clear alpha_temp;
                clear R_ip;
            end
        end
    end
    alpha_alpha = reshape(alpha_alpha,[m^2,m^2]);
    result = alpha_alpha;
    
else
    %% compute the RHS of equation
    % R1 for the equation for finding theta_theta
    % R2 for the equation for finding alpha_theta
    % R3 for the equation for finding alpha_alpha
    
    % Compute int_b
    int_b = fct_comp_int(param,bt);
    % Define Matrix M and N
    MM = fct_comp_mat_M(param,bt);
    NN = fct_comp_mat_N(param,MM,int_b);
    
    [R1,R2,R3] = fct_comp_RHS(param,bt,d_bt);
    
    % Compute alpha_theta
    alpha_theta = zeros(m,m,m);%(m_p,m_i,m_j)
    for j = 1:m
        for i = 1:m
            RHS = zeros(m,1);
            for p = 1:m
                RHS(p,1) = T*R2(p,i,j) - int_b(p)*R1(i,j);
            end
            alpha_theta_temp = linsolve(NN,RHS);
            alpha_theta(:,i,j) = alpha_theta_temp;
            clear RHS;
        end
    end
    
%     warning('trick');
%     alpha_theta=zeros(size(alpha_theta));
    
    % Compute theta_theta
    theta_theta = zeros(m,m);%(m_i,m_j)
    for i = 1:m
        for j = 1:m
            RSH1 = 0;
            for k = 1: m
                RSH1 = RSH1 + alpha_theta(k,i,j)*int_b(k);
            end
            RSH = R1(i,j) - RSH1;
            clear RSH1;
            theta_theta(i,j) = RSH/T;
            clear RSH;
        end
    end
    
%     warning('trick');
%     theta_theta=zeros(size(theta_theta));
    
    % COmpute alpha_alpha
    
    alpha_alpha = zeros(m,m,m,m);%(m_p,m_i,m_q,m_j)
    
    for j = 1:m
        for q = 1:m
            for i = 1:m
                R_ip = zeros(m,1);
                for p = 1:m
                    
                    R_ip(p,1) = R3(p,i,q,j) - int_b(p)*alpha_theta(i,q,j);
%                     R_ip(p,1) = R_ip(p,1) - R3(p,i,q,j) - int_b(p)*alpha_theta(i,q,j);
                end
                alpha_temp = linsolve(MM,R_ip);
                alpha_alpha(:,i,q,j) = alpha_temp;
            end
            
        end
    end
    
%     warning('trick');
%     alpha_alpha=zeros(size(alpha_alpha));
    
    alpha_alpha = reshape(alpha_alpha,[m^2,m^2]);
    alpha_theta = reshape(alpha_theta,[m^2,m]);
    
    result1 = [theta_theta;alpha_theta];
    result2 = [alpha_theta'; alpha_alpha];
    result = [result1,result2];
    clear result1;
    clear result2;
    
end

%% Force the symetry and the positivity of the matrix
result = 1/2*(result +result');
[V,D]=eig(result);
D=diag(D);
D(D<0)=0;
result=V*diag(D)*V';
end