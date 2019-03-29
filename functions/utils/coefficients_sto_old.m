function L = coefficients_sto(param)
% Compute advection and diffusion coefficients

%% Parameters of the program
m = param.nb_modes;
d = param.d;
MX = param.MX;
dX = param.dX;
M = MX(1)*MX(2);


%% Loading needed files for calculation
load([param.folder_data 'diffusion_mode_' param.type_data '_' num2str(m) '_modes_a_cst_threshold_0_0005.mat']);
load([param.folder_data 'mode_' param.type_data '_' num2str(m) '_modes.mat']);

%% Computing the first component
phi_m_U = phi_m_U(:,1:m,:); % M x m x d

phi_m_U = reshape(phi_m_U, [MX, m d]); % Mx x My x m x d
phi_m_U = permute(phi_m_U, [3 4 1 2]); % m x d x Mx x My

%compute differials
grad_phi = gradient_mat(phi_m_U, dX); % m x d x Mx x My x d'
grad_phi = permute(grad_phi, [3 4 5 2 1]); % Mx x My x d' x d x m

phi_m_U = permute(phi_m_U, [3 4 2 1]); % Mx x My x d x m

z = permute(z, [1 3 4 2]); % M x d x d
z = reshape(z, [MX d d]); % Mx x My x d x d
F1 = zeros(m);
for i=1:m
    for j=1:m
        div_f_temp = zeros(MX);
        
        xi = zeros([MX 1 d]);
        for kbis=1:d
            grad_phi_jk = grad_phi(:,:,:,kbis,j); % Mx x My x d
            dx_f1x = reshape(z(:,:,1,1).*grad_phi_jk(:,:,1) + z(:,:,1,2).*grad_phi_jk(:,:,2),[ 1 1 1 1 MX]);
            dy_f1y = reshape(z(:,:,2,1).*grad_phi_jk(:,:,1) + z(:,:,2,2).*grad_phi_jk(:,:,2),[ 1 1 1 1 MX]);
            dtot_f1 = permute(dx_f1x + dy_f1y, [5 6 1 2 3 4]); % Mx x My
            xi(:,:,kbis) = dtot_f1;
        end
        xi = reshape(xi, [M 1 d]); % M x 1 x d
        xi = proj_div_propre(xi, MX, dX); % M x 1 x d
        xi = reshape(xi, [MX d]); % Mx x My x d
        
        for k=1:d
            div_f_temp = div_f_temp + xi(:,:,k).*phi_m_U(:,:,k,i); % Mx x My
        end
        F1(j,i) = 0.5*integration_mat(reshape(div_f_temp, [1 1 MX]), dX);
    end
end
clear f_temp div_f_temp grad_phi_jk

%% Computing the second component
z = reshape(z, [MX 1 d d]); % Mx x My x 1 x d x d
% compute differials
dx_z = diff_l(z,1,dX); % Mx x My x 1 x d x d
dy_z = diff_l(z,2,dX); % Mx x My x 1 x d x d

div_z = zeros([MX d]);
div_z(:,:,1) = dx_z(:,:,1,1,1) + dy_z(:,:,1,2,1);
div_z(:,:,2) = dx_z(:,:,1,1,2) + dy_z(:,:,1,2,2);

F2 = zeros(m);
for i=1:m
    for j=1:m
        xi = zeros([MX 1 d]);
        for kbis=1:d
            xi(:,:,kbis) = div_z(:,:,1).*grad_phi(:,:,1,kbis,j) + div_z(:,:,2).*grad_phi(:,:,2,kbis,j);
        end
        xi = reshape(xi, [M 1 d]); % M x 1 x d
        xi = proj_div_propre(xi, MX, dX); % M x 1 x d
        xi = reshape(xi, [MX d]); % Mx x My x d
        
        for k=1:d
            F2(j,i) = F2(j,i) + 0.5 * integration_mat(reshape(xi(:,:,k).*phi_m_U(:,:,k,i), [1 1 MX]), dX);
        end
    end
end

L = F1 + F2;
end