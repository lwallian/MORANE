function [R1, R2] = fct_general_correlated_RHS(param, bt, d2bt, sigma_ss)
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here

dt = param.dt;
N_tot = param.N_tot;
m = param.nb_modes;
nu = param.viscosity;
T = N_tot*dt;
M = param.M;
dX = param.dX;
MX = param.MX;
d = param.d;
lambda = param.lambda;

% The last two time steps are not used
% N_tot = N_tot - 2;
% T = T - 2 * dt;

% Estimate the terms using the orthogonality assumptions
[R1, R2] = fct_comp_correlated_RHS(param, bt, d2bt);

% As they are no longer valid, the statistics are going to be estimated
% through least squares
beta = bsxfun(@times, R1, 1 / lambda);

% Compute gamma
% First, we define G_pq
G_pq = mean(bt' * bt, 1); % check that it's the outer product given the dimensions, should be nxn

% We define psi_p
psi = zeros(M, m, d); % vector in space and we reshape later on

% Initialization
t_local = 1; % index of the snapshot in a file
if param.data_in_blocks.bool % if data are saved in several files
    big_T = 1; % index of the file
    name_file_U_temp=param.name_file_U_temp{big_T}; % Name of the next file
else
    name_file_U_temp=param.name_file_U_temp; % Name of the next file
end

load(name_file_U_temp);

for t = 1 : N_tot % loop on time
    if t_local > size(U, 2) % A new file needs to be loaded
        % Save previous file with residual velocity
        save(name_file_U_temp, 'U', '-v7.3');
        % initialization of the index of the snapshot in the file
        t_local = 1;
        % Incrementation of the file index
        big_T = big_T + 1;
        % Name of the new file
        name_file_U_temp = param.name_file_U_temp{big_T};
        % Load new file
        load(name_file_U_temp);
    end
    for k = 1 : d
        for p = 1 : m + 1
            if p ~= m + 1
                psi(:, p, k) = psi(:, p, k) + bt(t_local, p) .* U(:, t_local, k);
            else
                psi(:, p, k) = psi(:, p, k) + U(:, t_local, k);
            end
        end
    end
    clear U;
    t_local = t_local + 1;
end
psi = psi ./ T;

% Estimate gamma with sigma_ss and projecting over psi
gamma = zeros(m, m, m, m);

for t = 1 : T
    B_dot = randn(m, m + 1);
    R = operator_R(sigma_ss * B_dot, psi);
    Q = operator_Q(sigma_ss * B_dot, phi);
    for p = 1 : m + 1
        for i = 1 : m + 1
            for q = 1 : m + 1
                for j = 1 : m + 1
                    gamma(p, i, q, j) = gamma(p, i, q, j) + R(p, i) + Q(p, i);
                end
            end
        end
    end
    clear B_dot R Q;
end
gamma = gamma .* dt / T;

% Use Least Squares to estimate the theta_theta in the general case
%% TODO

end

function R = operator_R(xi, psi)

R = zeros(m + 1, m + 1);

% Estimate the xi's gradient
xi = permute(xi, [3, 4, 1, 2]);%(1,d,M)
xi = reshape(xi, [1, d, MX]);%(1,d,Mx,My,(Mz))
xi = permute(xi, [ndims(xi) + 1, 1 : ndims(xi)]);%(1,1,d,Mx ,My,(Mz))

dxi = gradient_mat(xi);
dxi = permute(dxi, [ndims(dxi) + 1, 1, ndims(dxi), 3 : ndims(dxi) - 1, 2]);

% Do the projection for each q and i
for q = 1 : m + 1
    % Small scale advected by large scale
    psi_q = psi(:, q, :);%(M,1,d)
    psi_q = permute(psi_q, [2, 3, 1]);%(1,d,M)
    psi_q = reshape(psi_q, [1, d, MX]);%(1,d,Mx,My,(Mz))
    dpsi_q = gradient_mat(psi_q, dX);
    dpsi_q = permute(dpsi_q, [ndims(dpsi_q) + 1, 1, ndims(dpsi_q), 3 : ndims(dpsi_q) - 1, 2]);
    %(1,1,d!,Mx,My,(Mz),d
    
    adv_sl = bsxfun(@times, dpsi_q, xi); % 1 x (1) x d x Mx x My (x Mz) d
    clear dphi_q;
    adv_sl = sum(adv_sl, 3);
    adv_sl = permute(adv_sl, [1, 2, 4 : ndims(adv_sl), 3]);%(1 1 Mx My (Mz) d)
    
    % Large scale advected by small scale
    adv_ls = bsxfun(@times, dxi, psi_q);
    clear psi_q;
    adv_ls = sum(adv_ls, 3);
    adv_ls = permute(adv_ls, [1, 2, 4 : ndims(adv_ls), 3]);%(1 1 Mx My (Mz) d)
    
    % projection on phi_j
    for j = 1 : m + 1
        psi_j = psi(:, j, :);
        psi_j = permute(psi_j, [4, 2, 1, 3]);%(1,1,M,d)
        psi_j = reshape(psi_j, [1, 1, MX, d]);%(1,1,Mx,My,(Mz),d)
        
        s_temp = adv_sl .* psi_j + adv_ls .* psi_j; %(1,1,Mx,My,(Mz),d)
        clear psi_j;
        s_temp = sum(s_temp, ndims(s_temp));%(1,1,Mx,My,(Mz))
        
        % compute the integration of s_temp
        s_temp = integration_mat(s_temp, dX);
        R(q,j) = - s_temp;
        
        clear s_temp;
    end
    clear adv_sl adv_ls;
end

end

function Q = operator_Q(xi, phi)

Q = zeros(m + 1, m + 1);

% Estimate the xi's gradient
xi = permute(xi, [3, 4, 1, 2]);%(1,d,M)
xi = reshape(xi, [1, d, MX]);%(1,d,Mx,My,(Mz))
xi = permute(xi, [ndims(xi) + 1, 1 : ndims(xi)]);%(1,1,d,Mx ,My,(Mz))

dxi = gradient_mat(xi);
dxi = permute(dxi, [ndims(dxi) + 1, 1, ndims(dxi), 3 : ndims(dxi) - 1, 2]);

% Do the projection for each q and i
for q = 1 : m + 1
    % Small scale advected by large scale
    phi_q = phi(:, q, :);%(M,1,d)
    phi_q = permute(phi_q, [2, 3, 1]);%(1,d,M)
    phi_q = reshape(phi_q, [1, d, MX]);%(1,d,Mx,My,(Mz))
    dphi_q = gradient_mat(phi_q, dX);
    dphi_q = permute(dphi_q, [ndims(dphi_q) + 1, 1, ndims(dphi_q), 3 : ndims(dphi_q) - 1, 2]);
    %(1,1,d!,Mx,My,(Mz),d
    
    adv_sl = bsxfun(@times, dphi_q, xi); % 1 x (1) x d x Mx x My (x Mz) d
    clear dphi_q;
    adv_sl = sum(adv_sl, 3);
    adv_sl = permute(adv_sl, [1, 2, 4 : ndims(adv_sl), 3]);%(1 1 Mx My (Mz) d)
    
    % Large scale advected by small scale
    adv_ls = bsxfun(@times, dxi, phi_q);
    clear psi_q;
    adv_ls = sum(adv_ls, 3);
    adv_ls = permute(adv_ls, [1, 2, 4 : ndims(adv_ls), 3]);%(1 1 Mx My (Mz) d)
    
    % Add the diffusion term if phi_0
    if q == m + 1
        Lap_xi = laplacian_mat(xi, dX);
        Lap_xi = nu*Lap_xi;
        Lap_xi = permute(Lap_xi,[1 ndims(Lap_xi)+1 2:ndims(Lap_xi)]);
        Lap_xi = permute(Lap_xi, [1 2 4:ndims(Lap_xi) 3]);%(1,1,Mx,My,(Mz),d)
    end
    
    % projection on phi_j
    for j = 1 : m + 1
        phi_j = phi(:, j, :);
        phi_j = permute(phi_j, [4, 2, 1, 3]);%(1,1,M,d)
        phi_j = reshape(phi_j, [1, 1, MX, d]);%(1,1,Mx,My,(Mz),d)
        
        if q == m + 1
            s_temp = adv_sl .* phi_j + adv_ls .* phi_j - Lap_xi; %(1,1,Mx,My,(Mz),d)
        else
            s_temp = adv_sl .* phi_j + adv_ls .* phi_j;
        end
        clear psi_j;
        s_temp = sum(s_temp, ndims(s_temp));%(1,1,Mx,My,(Mz))
        
        % compute the integration of s_temp
        s_temp = integration_mat(s_temp, dX);
        Q(q,j) = - s_temp;
        
        clear s_temp;
    end
    clear adv_sl adv_ls Lap_xi;
end

end

