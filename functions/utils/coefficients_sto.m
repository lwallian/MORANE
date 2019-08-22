function [F1 F2] = coefficients_sto(param)
% Compute advection and diffusion coefficients

%% Parameters of the program
m = param.nb_modes;
d = param.d;
MX = param.MX;
dX = param.dX;
M = prod(MX);


%% Loading needed files for calculation
load(param.name_file_diffusion_mode);
load(param.name_file_mode);
% load([param.folder_data 'diffusion_mode_' param.type_data '_' num2str(m) '_modes_a_cst_threshold_0_0005.mat']);
% load([param.folder_data 'mode_' param.type_data '_' num2str(m) '_modes.mat']);

%% Computing the first component
phi_m_U = phi_m_U(:,1:m,:); % M x m x d

phi_m_U = reshape(phi_m_U, [MX, m d]); % Mx x My (x Mz) x m x d
if d == 2
    phi_m_U = permute(phi_m_U, [3 4 1 2]); % m x d x Mx x My
else
    phi_m_U = permute(phi_m_U, [4 5 1 2 3]); % m x d x Mx x My (x Mz)
end


%compute differials
grad_phi = gradient_mat(phi_m_U, dX); % m x d x Mx x My (x Mz) x d'

if d == 2
    grad_phi = permute(grad_phi, [3 4 5 2 1]); % Mx x My x d' x d x m
    phi_m_U = permute(phi_m_U, [3 4 1 2]); % Mx x My x m x d
else
    grad_phi = permute(grad_phi, [3 4 5 6 2 1]); % Mx x My x Mz x d' x d x m
    phi_m_U = permute(phi_m_U, [3 4 5 1 2]); % Mx x My x Mz x m x d
end

phi_m_U = reshape(phi_m_U, [M m d]); % M x m x d


z = reshape(z, [MX d d]); % Mx x My (x Mz) x d x d


F1 = zeros(m);
if d == 2
    for i=1:m
        for j=1:m
            xi_1 = zeros([MX 1 d]);
            %computing components of xi_1
            for l=1:d
                grad_phi_jl = grad_phi(:,:,:,l,j); % Mx x My x d'
                z_gradphi = zeros([MX d]);
                z_gradphi(:,:,1) = z(:,:,1,1).*grad_phi_jl(:,:,1) + z(:,:,1,2).*grad_phi_jl(:,:,2); % Mx x My
                z_gradphi(:,:,2) = z(:,:,2,1).*grad_phi_jl(:,:,1) + z(:,:,2,2).*grad_phi_jl(:,:,2); % Mx x My
                z_gradphi = permute(z_gradphi, [3 1 2]); % d x Mx x My
                z_gradphi = reshape(z_gradphi, [ 1 1 d 1 MX]); % 1 x 1 x d x 1 x Mx x My
                dx_z_gradphi = diff_l(z_gradphi,1,dX); % 1 x 1 x d x 1 x Mx x My
                dy_z_gradphi = diff_l(z_gradphi,2,dX); % 1 x 1 x d x 1 x Mx x My
                div_z_gradphi = dx_z_gradphi(1,1,1,1,:,:) + dy_z_gradphi(1,1,2,1,:,:); % 1 x 1 x 1 x 1 x Mx x My
                div_z_gradphi = permute(div_z_gradphi, [5 6 1 2 3 4]); % Mx x My
                xi_1(:,:,1,l) = div_z_gradphi;
            end
            
            xi_1 = reshape(xi_1, [ M 1 d]); % M x 1 x d
            proj_xi_1 = xi_1 - proj_div_propre(xi_1, MX, dX); % M x 1 x d
            
            f1_ji = sum(phi_m_U(:,i,:) .* proj_xi_1,3); % M x 1
            f1_ji = 0.5*reshape(f1_ji, [ 1 1 MX]); % 1 x 1 x Mx x My
            F1(j,i) = integration_mat(f1_ji, dX);
        end
    end
    
else
    for i=1:m
        for j=1:m
            xi_1 = zeros([MX 1 d]);
            %computing components of xi_1
            for l=1:d
                grad_phi_jl = grad_phi(:,:,:,:,l,j); % Mx x My x Mz x d'
                z_gradphi = zeros([MX d]);
                
                z_gradphi(:,:,:,1) = z(:,:,:,1,1).*grad_phi_jl(:,:,:,1) + z(:,:,:,1,2).*grad_phi_jl(:,:,:,2) + z(:,:,:,1,3).*grad_phi_jl(:,:,:,3); % Mx x My x Mz
                z_gradphi(:,:,:,2) = z(:,:,:,2,1).*grad_phi_jl(:,:,:,1) + z(:,:,:,2,2).*grad_phi_jl(:,:,:,2) + z(:,:,:,2,3).*grad_phi_jl(:,:,:,3); % Mx x My x Mz
                z_gradphi(:,:,:,2) = z(:,:,:,3,1).*grad_phi_jl(:,:,:,1) + z(:,:,:,3,2).*grad_phi_jl(:,:,:,2) + z(:,:,:,3,3).*grad_phi_jl(:,:,:,3); % Mx x My x Mz
                
                z_gradphi = permute(z_gradphi, [4 1 2 3]); % d x Mx x My x Mz
                z_gradphi = reshape(z_gradphi, [ 1 1 d 1 MX]); % 1 x 1 x d x 1 x Mx x My x Mz
                dx_z_gradphi = diff_l(z_gradphi,1,dX); % 1 x 1 x d x 1 x Mx x My x Mz
                dy_z_gradphi = diff_l(z_gradphi,2,dX); % 1 x 1 x d x 1 x Mx x My x Mz
                dz_z_gradphi = diff_l(z_gradphi,3,dX); % 1 x 1 x d x 1 x Mx x My x Mz
                div_z_gradphi = dx_z_gradphi(1,1,1,1,:,:,:) + dy_z_gradphi(1,1,2,1,:,:,:) + dz_z_gradphi(1,1,3,1,:,:,:); % 1 x 1 x 1 x 1 x Mx x My x Mz
                div_z_gradphi = permute(div_z_gradphi, [5 6 7 1 2 3 4]); % Mx x My x Mz
                xi_1(:,:,:,1,l) = div_z_gradphi;
            end
            
            xi_1 = reshape(xi_1, [ M 1 d]); % M x 1 x d
            proj_xi_1 = xi_1 - proj_div_propre(xi_1, MX, dX); % M x 1 x d
            
            f1_ji = sum(phi_m_U(:,i,:) .* proj_xi_1,3); % M x 1
            f1_ji = 0.5*reshape(f1_ji, [ 1 1 MX]); % 1 x 1 x Mx x My x Mz
            F1(j,i) = integration_mat(f1_ji, dX);
        end
    end
end
%% Computing the second component
z = reshape(z, [MX 1 d*d]); % Mx x My (x Mz) x 1 x d^2
if d == 2
    z = permute(z, [4 1 2 3]); % d^2 x Mx x My
else
    z = permute(z, [5 1 2 3 4]); % d^2 x Mx x My x Mz
end
z = reshape(z, [1 1 d*d 1 MX]); % 1 x 1 x d^2 x 1 x Mx x My (x Mz)
% compute differials
dx_z = diff_l(z,1,dX); % 1 x 1 x d^2 x 1 x Mx x My (x Mz)
dy_z = diff_l(z,2,dX); % 1 x 1 x d^2 x 1 x Mx x My (x Mz)
if d == 2
    dx_z = permute(dx_z, [5 6 3 1 2 4]); % Mx x My x d^2
    dy_z = permute(dy_z, [5 6 3 1 2 4]); % Mx x My x d^2
else
    dx_z = permute(dx_z, [5 6 7 3 1 2 4]); % Mx x My x Mz x d^2
    dy_z = permute(dy_z, [5 6 7 3 1 2 4]); % Mx x My x Mz x d^2
    
    dz_z = diff_l(z,3,dX); % 1 x 1 x d^2 x 1 x Mx x My x Mz
    dz_z = permute(dz_z, [5 6 7 3 1 2 4]); % Mx x My x d^2
    dz_z = reshape(dz_z, [MX d d]); % Mx x My x Mz x d x d
end

dx_z = reshape(dx_z, [MX d d]); % Mx x My (x Mz) x d x d
dy_z = reshape(dy_z, [MX d d]); % Mx x My (x Mz) x d x d

div_z = zeros([MX d]);
if d == 2
div_z(:,:,1) = dx_z(:,:,1,1) + dy_z(:,:,2,1);
div_z(:,:,2) = dx_z(:,:,1,2) + dy_z(:,:,2,2);
else
    div_z(:,:,:,1) = dx_z(:,:,:,1,1) + dy_z(:,:,:,2,1) + dz_z(:,:,:,3,1);
    div_z(:,:,:,2) = dx_z(:,:,:,1,2) + dy_z(:,:,:,2,2) + dz_z(:,:,:,3,2);
    div_z(:,:,:,3) = dx_z(:,:,:,1,3) + dy_z(:,:,:,2,3) + dz_z(:,:,:,3,3);
end

F2 = zeros(m);

for i=1:m
    for j=1:m
        xi_2 = zeros([MX 1 d]);
        for l=1:d
            if d == 2
            grad_phi_jl = grad_phi(:,:,:,l,j); % Mx x My x d'
            xi_2(:,:,1,l) = div_z(:,:,1).*grad_phi_jl(:,:,1) + ...
                div_z(:,:,2).*grad_phi_jl(:,:,2); % Mx x My
            else
                grad_phi_jl = grad_phi(:,:,:,:,l,j); % Mx x My x Mz x d'
            xi_2(:,:,:,1,l) = div_z(:,:,:,1).*grad_phi_jl(:,:,:,1) + ...
                div_z(:,:,:,2).*grad_phi_jl(:,:,:,2) + ...
                div_z(:,:,:,3).*grad_phi_jl(:,:,:,3); % Mx x My x Mz
            end
        end
        
        xi_2 = reshape(xi_2, [ M 1 d]); % M x 1 x d
        proj_xi_2 = xi_2 - proj_div_propre(xi_2, MX, dX); % M x 1 x d
        
        f2_ji = sum(phi_m_U(:,i,:) .* proj_xi_2,3); % M x 1
        f2_ji = 0.5*reshape(f2_ji, [ 1 1 MX]); % 1 x 1 x Mx x My (x Mz)
        F2(j,i) = integration_mat(f2_ji, dX);
    end
end

F1 = -F1;
F2 = -F2;
end