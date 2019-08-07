function [R1, R2, R3] = fct_general_correlated_RHS(param, bt, d2bt, sigma_ss)
%FCT_COMP_CORRELATED_RHS Estimates the noise statistics in the correlated
%non resolved modes scheme given the PCA residual and the chronos functions
%   @param param: structure with lots of parameters concerning the current
%   simulation
%   @param bt: chronos function of the current model
%   @param d2bt: second derivative (wrt time) of the aforementioned chronos
%   function
%   @param sigma_ss: noise correlation matrix for the non resolved modes
%   @return R1: value proportional to the theta_theta term
%   @return R2: value proportional to the Mi_sigma_sigma term
%   @return R3: value proportional to the xi_xi_inf term
%
% Author: Agustin PICARD, intern @ Scalian with Valentin RESSEGUIER as
% supervisor
%

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

% Estimate the terms using the orthogonality assumptions
[R1, R2, ~] = fct_comp_correlated_RHS(param, bt, d2bt);

% As they are no longer valid, the statistics are going to be estimated
% through least squares
beta = R1 ./ lambda;

% Compute gamma
% First, we define G_pq
G_pq = bt(1 : end - 1, :)' * bt(1 : end - 1, :); % check that it's the outer product given the dimensions, should be nxn

% We define psi_p
psi = zeros(M, m, d); % vector in space and we reshape later on

% Estimate the psi modes for projections later on, calculated from the
% residual of the velocity field
t_local = 1; % index of the snapshot in a file
if param.data_in_blocks.bool % if data are saved in several files
    big_T = 1; % index of the file
    name_file_U_temp=param.name_file_U_temp{big_T}; % Name of the next file
else
    name_file_U_temp=param.name_file_U_temp; % Name of the next file
end

load(name_file_U_temp, 'U');

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
        load(name_file_U_temp, 'U');
    end
    for k = 1 : d
        for p = 1 : m
            psi(:, p, k) = psi(:, p, k) + bt(t_local, p) .* U(:, t_local, k);
        end
    end
    t_local = t_local + 1;
end
psi = psi ./ T;

% Estimate gamma with sigma_ss and projecting over psi
gamma = zeros(m, m, m + 1, m + 1);
load(param.name_file_mode, 'phi_m_U')

for t = 1 : T
    B_dot = prod(MX * d) .* randn([MX, d]);
    xi = sigma_ss .* B_dot;
    R = operator_R(xi, psi, param);
    Q = operator_Q(xi, phi_m_U, param);
    for p = 1 : m
        for i = 1 : m
            for q = 1 : m + 1
                for j = 1 : m + 1
                    gamma(p, i, q, j) = gamma(p, i, q, j) + R(p, i) * Q(q, j);
                end
            end
        end
    end
    clear B_dot R Q;
end
gamma = gamma * dt;

% Use Least Squares to estimate the theta_theta in the general case
% G_pinv = pinv(G_pq);
R1 = zeros(m, m, m + 1, m + 1);
for i = 1 : m
    for q = 1 : m + 1
        for j = 1 : m + 1
            kappa = beta(:, i, q, j) - gamma(:, i, q, j);
            R1(:, i, q, j) = linsolve(G_pq, kappa); % more efficient than solving with pinv
%             R1(:, i, q, j) = G_pinv * kappa;
        end
    end
end

% Estimate the xi_xi_inf in the general case
R3 = zeros(m, m);
zeta = zeros(m, m);
for i = 1 : m
    for j = 1 : m
        for p = 1 : m
            for q = 1 : m
                zeta(i, j) = zeta(i, j) + G_pq(p, q) * R1(p, i, q, j);
            end
        end
    end
end
for i = 1 : m
    for j = 1 : m
        R3(i, j) = d2bt(:, i)' * d2bt(:, j) - zeta(i, j);
    end
end

end

function R = operator_R(xi, psi, param)

dt = param.dt;
N_tot = param.N_tot;
m = param.nb_modes;
M = param.M;
dX = param.dX;
MX = param.MX;
d = param.d;

R = zeros(m + 1, m + 1);

% Estimate the xi's gradient
xi = permute(xi, [3, 4, 1, 2]);%(1,d,M)
xi = reshape(xi, [1, d, MX]);%(1,d,Mx,My,(Mz))

dxi = gradient_mat(xi, dX);
dxi = permute(dxi, [ndims(dxi) + 1, 1, ndims(dxi), 3 : ndims(dxi) - 1, 2]);
xi = permute(xi, [ndims(xi) + 1, 1 : ndims(xi)]);%(1,1,d,Mx ,My,(Mz))

% Do the projection for each q and i
for q = 1 : m
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
    psi_q = permute(psi_q, [ndims(psi_q) + 1, 1 : ndims(psi_q)]);
    adv_ls = bsxfun(@times, dxi, psi_q);
    clear psi_q;
    adv_ls = sum(adv_ls, 3);
    adv_ls = permute(adv_ls, [1, 2, 4 : ndims(adv_ls), 3]);%(1 1 Mx My (Mz) d)
    
    % Do the projection onto the divergence free space
    integ = adv_sl + adv_ls;
    integ = permute(integ, [3 : ndims(integ) - 1, 1, 2, ndims(integ)]); % [Mx, My, (Mz), 1, 1, d]
    integ = reshape(integ, [M, 1, d]);
    if strcmp(param.type_data, 'turb2D_blocks_truncated')
        integ = integ - proj_div_propre(integ, MX, dX, true);
    else
        integ = integ - proj_div_propre(integ, MX, dX, false);
    end
    integ = reshape(integ, [MX, 1, d]);
    integ = permute(integ, [ndims(integ) - 1, 1 : ndims(integ) - 2, ndims(integ)]);
    integ = reshape(integ, [1, 1, MX, d]);
    
    % projection on phi_j
    for j = 1 : m
        psi_j = psi(:, j, :);
        psi_j = permute(psi_j, [4, 2, 1, 3]);%(1,1,M,d)
        psi_j = reshape(psi_j, [1, 1, MX, d]);%(1,1,Mx,My,(Mz),d)
        
        s_temp = integ .* psi_j; %(1,1,Mx,My,(Mz),d)
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

function Q = operator_Q(xi, phi, param)

m = param.nb_modes;
nu = param.viscosity;
M = param.M;
dX = param.dX;
MX = param.MX;
d = param.d;

Q = zeros(m + 1, m + 1);

% Estimate the xi's gradient
xi = permute(xi, [3, 4, 1, 2]);%(1,d,M)
xi = reshape(xi, [1, d, MX]);%(1,d,Mx,My,(Mz))

dxi = gradient_mat(xi, dX);
dxi = permute(dxi, [ndims(dxi) + 1, 1, ndims(dxi), 3 : ndims(dxi) - 1, 2]);
xi = permute(xi, [ndims(xi) + 1, 1 : ndims(xi)]);%(1,1,d,Mx ,My,(Mz))

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
    phi_q = permute(phi_q, [ndims(phi_q) + 1, 1 : ndims(phi_q)]);
    adv_ls = bsxfun(@times, dxi, phi_q);
    clear psi_q;
    adv_ls = sum(adv_ls, 3);
    adv_ls = permute(adv_ls, [1, 2, 4 : ndims(adv_ls), 3]);%(1 1 Mx My (Mz) d)
    
    % Add the diffusion term if phi_0
    if q == m + 1
        xi = reshape(xi, [1, d, MX]);
        Lap_xi = laplacian_mat(xi, dX);
        Lap_xi = nu*Lap_xi;
        Lap_xi = permute(Lap_xi,[1 ndims(Lap_xi)+1 2:ndims(Lap_xi)]);
        Lap_xi = permute(Lap_xi, [1 2 4:ndims(Lap_xi) 3]);%(1,1,Mx,My,(Mz),d)
        xi = permute(xi, [ndims(xi) + 1, 1 : ndims(xi)]);%(1,1,d,Mx ,My,(Mz))
    end
    
    % Do the divergence free projection
    if q ~= m + 1
        integ = adv_sl + adv_ls;
    else
        integ = adv_sl + adv_ls - Lap_xi;
    end
    integ = permute(integ, [3 : ndims(integ) - 1, 1, 2, ndims(integ)]); % [Mx, My, (Mz), 1, 1, d]
    integ = reshape(integ, [M, 1, d]);
    if strcmp(param.type_data, 'turb2D_blocks_truncated')
        integ = integ - proj_div_propre(integ, MX, dX, true);
    else
        integ = integ - proj_div_propre(integ, MX, dX, false);
    end
    integ = reshape(integ, [MX, 1, d]);
    integ = permute(integ, [ndims(integ) - 1, 1 : ndims(integ) - 2, ndims(integ)]);
    integ = reshape(integ, [1, 1, MX, d]);
    
    % projection on phi_j
    for j = 1 : m + 1
        phi_j = phi(:, j, :);
        phi_j = permute(phi_j, [4, 2, 1, 3]);%(1,1,M,d)
        phi_j = reshape(phi_j, [1, 1, MX, d]);%(1,1,Mx,My,(Mz),d)
        
        s_temp = integ .* phi_j; %(1,1,Mx,My,(Mz),d)
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

