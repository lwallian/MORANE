function [R1, R2, R3, C1, C2] = fct_comp_correlated_RHS(param, bt, d2bt)
%FCT_COMP_CORRELATED_RHS Estimates the noise statistics in the correlated
%non resolved modes scheme given the PCA residual and the chronos functions
%   @param param: structure with lots of parameters concerning the current
%   simulation
%   @param bt: chronos function of the current model
%   @param d2bt: second derivative (wrt time) of the aforementioned chronos
%   function
%   @return R1: value proportional to the theta_theta term (theta_theta * T)
%   @return R2: value proportional to the Mi_sigma_sigma term (Mi_sigma_sigma * T)
%   @return R3: value proportional to the xi_xi_inf term (xi_xi_inf * T)
%   @return C1: initial value for Mi_ss
%   @return C2: initial value for eta
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

% The last two time steps are not used
N_tot = N_tot - 2;
T = T - 2 * dt;

%% load U

% Initialization
t_local = 1; % index of the snapshot in a file
if param.data_in_blocks.bool % if data are saved in several files
    big_T = 1; % index of the file
    name_file_U_temp=param.name_file_U_temp{big_T}; % Name of the next file
else
    name_file_U_temp=param.name_file_U_temp; % Name of the next file
end

load(name_file_U_temp, 'U');
dU = diff(U, 1, 2);

% compute the sum((dw_ss * del) * dw_ss)
Mi_sigma = zeros(M, d);
%compute the sum(b_p * d2bt_i * dX_res)
del_pi = zeros(M, m + 1, m, d); % (M,m(p),m(i),d)
% add the constant term to bt
bt = cat(2, bt, ones(size(bt,1), 1));
lambda(m + 1) = T.^3 / 3 / T;

if iscell(param.name_file_U_temp)
    N_final = 2 * length(param.name_file_U_temp);
else
    N_final = 2;
end
for t = 1 : N_tot - N_final % loop in time
    if t_local > size(U, 2) - 2 % A new file needs to be loaded
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
        
        % Differentiate it wrt time
        dU = diff(U, 1, 2);
    end
    % Calculate (dw_ss * del) * dw_ss
    dU_del_dU = calculate_dU_del_dU(dU(:, t_local, :), d, MX, dX); % [1, 1, Mx, My(, Mz), d]
    dU_del_dU = squeeze(dU_del_dU);
    dU_del_dU = reshape(dU_del_dU, [M, d]);
    for k = 1 : d
        Mi_sigma(:, k) = Mi_sigma(:, k) + dU_del_dU(:, k);
        for i = 1 : m
            for p = 1 : m + 1
                del_pi(:, p, i, k) = del_pi(:, p, i, k) ...
                    + dU(:, t_local, k) * d2bt(t_local, i) ...
                    * bt(t_local, p) / lambda(p);
            end
        end
    end
    % Incrementation of the index of the snapshot in the file
    t_local = t_local + 1;
end
C2_mat = calculate_dU_del_dU(dU(:, end, :), d, MX, dX);
clear dU dU_del_dU;
clear U;

% Do the divergence free projection before projecting onto each topos
Mi_sigma = reshape(Mi_sigma, [M, 1, d]);
C2_mat = reshape(C2_mat, [M, 1, d]);
if param.eq_proj_div_free == 2
    if strcmp(param.type_data, 'turb2D_blocks_truncated')
        Mi_sigma = Mi_sigma - proj_div_propre(Mi_sigma, MX, dX, true);
        C2_mat = C2_mat - proj_div_propre(C2_mat, MX, dX, true);
    else
        Mi_sigma = Mi_sigma - proj_div_propre(Mi_sigma, MX, dX, false);
        C2_mat = C2_mat - proj_div_propre(C2_mat, MX, dX, false);
    end
end

% Load the topos
load(param.name_file_mode, 'phi_m_U')

% Compute the projection over each mode for Mi_sigma
R2 = zeros(m, 1);
C2 = zeros(m, 1);
Mi_sigma = reshape(Mi_sigma, [1, 1, MX, d]);
C2_mat = reshape(C2_mat, [1, 1, MX, d]);
for j = 1 : m
    phi_j = phi_m_U(:, j, :);
    phi_j = permute(phi_j, [4, 2, 1, 3]);%(1,1,M,d)
    phi_j = reshape(phi_j, [1, 1, MX, d]);%(1,1,Mx,My,(Mz),d)
    
    s_temp = Mi_sigma .* phi_j; %(1,1,Mx,My,(Mz),d)
    c_temp = C2_mat .* phi_j;
    clear phi_j;
    s_temp = sum(s_temp, ndims(s_temp));%(1,1,Mx,My,(Mz))
    c_temp = sum(c_temp, ndims(c_temp));
    
    % compute the integration of s_temp
    s_temp = integration_mat(s_temp, dX);
    c_temp = integration_mat(c_temp, dX);
    R2(j) = -s_temp;
    C2(j) = -c_temp;
    clear s_temp c_temp;
end

% Compute theta_theta:
R1 = zeros(m + 1, m, m + 1, m);

for p = 1 : m + 1
    for i = 1 : m
        del_p_i = del_pi(:, p, i, :);
        del_p_i = permute(del_p_i, [3, 4, 1, 2]);%(1,d,M)
        del_p_i = reshape(del_p_i, [1, d, MX]);%(1,d,Mx,My,(Mz))
        
        ddel_p_i = gradient_mat(del_p_i, dX);
        ddel_p_i = permute(ddel_p_i, [ndims(ddel_p_i) + 1, 1, ndims(ddel_p_i), 3 : ndims(ddel_p_i) - 1, 2]);
        del_p_i = permute(del_p_i, [ndims(del_p_i) + 1, 1 : ndims(del_p_i)]);%(1,1,d,Mx ,My,(Mz))
        
        % compute the gradient of phi_q
        for q = 1 : m + 1
            % Small scale advected by large scale
            phi_q = phi_m_U(:, q, :);%(M,1,d)
            phi_q = permute(phi_q, [2, 3, 1]);%(1,d,M)
            phi_q = reshape(phi_q, [1, d, MX]);%(1,d,Mx,My,(Mz))
            dphi_q = gradient_mat(phi_q, dX);
            dphi_q = permute(dphi_q, [ndims(dphi_q) + 1, 1, ndims(dphi_q), 3 : ndims(dphi_q) - 1, 2]);
                    %(1,1,d!,Mx,My,(Mz),d
            
            % compute the advection term_piq ::adv_piq
            adv_sl = bsxfun(@times, del_p_i, dphi_q); % 1 x (1) x d x Mx x My (x Mz) d
            clear dphi_q;
            adv_sl = sum(adv_sl, 3);
            adv_sl = permute(adv_sl, [1, 2, 4 : ndims(adv_sl), 3]);%(1 1 Mx My (Mz) d)
            
            % Large scale advected by small scale
            phi_q = permute(phi_q, [ndims(phi_q) + 1, 1 : ndims(phi_q)]);
            adv_ls = bsxfun(@times, phi_q, ddel_p_i);
            adv_ls = sum(adv_ls, 3);
            adv_ls = permute(adv_ls, [1, 2, 4 : ndims(adv_ls), 3]);%(1 1 Mx My (Mz) d)
            
            % Do the divergence free projection
            integ = adv_sl + adv_ls;
            integ = permute(integ, [3 : ndims(integ) - 1, 1, 2, ndims(integ)]); % [Mx, My, (Mz), 1, 1, d]
            integ = reshape(integ, [M, 1, d]);
            if param.eq_proj_div_free == 2
                if strcmp(param.type_data, 'turb2D_blocks_truncated')
                    integ = integ - proj_div_propre(integ, MX, dX, true);
                else
                    integ = integ - proj_div_propre(integ, MX, dX, false);
                end
            end
            integ = reshape(integ, [MX, 1, d]);
            integ = permute(integ, [ndims(integ) - 1, 1 : ndims(integ) - 2, ndims(integ)]);
            integ = reshape(integ, [1, 1, MX, d]);
            
            % Add the diffusion term if phi_0
            if q == m + 1
                del_p_i = reshape(del_p_i, [1, d, MX]);
                Lap_del_pi = laplacian_mat(del_p_i,dX);
                Lap_del_pi = nu*Lap_del_pi;
                Lap_del_pi = permute(Lap_del_pi,[1 ndims(Lap_del_pi)+1 2:ndims(Lap_del_pi)]);
                Lap_del_pi = permute(Lap_del_pi, [1 2 4:ndims(Lap_del_pi) 3]);%(1,1,Mx,My,(Mz),d)
                del_p_i = permute(del_p_i, [ndims(del_p_i) + 1, 1 : ndims(del_p_i)]);%(1,1,d,Mx ,My,(Mz))
                
                integ = integ - Lap_del_pi;
            end
            
            % projection on phi_j
            for j = 1 : m
                phi_j = phi_m_U(:, j, :);
                phi_j = permute(phi_j, [4, 2, 1, 3]);%(1,1,M,d)
                phi_j = reshape(phi_j, [1, 1, MX, d]);%(1,1,Mx,My,(Mz),d)
                    
                s_temp = integ .* phi_j; %(1,1,Mx,My,(Mz),d)
                clear phi_j;
                s_temp = sum(s_temp, ndims(s_temp));%(1,1,Mx,My,(Mz))
                
                % compute the integration of s_temp
                s_temp = integration_mat(s_temp, dX);
                R1(p,i,q,j) = - s_temp;
                
                clear s_temp;
            end
            clear adv_sl adv_ls Lap_del_pi;
        end
        clear del_p_i ddel_p_i;
    end
end
clear del_pi;

% Compute xi_xi_inf
R3 = zeros(m, m);

for i = 1 : m
    for j = 1 : m
        lambda_theta_theta = 0;
        for k = 1 : m + 1
            lambda_theta_theta = lambda_theta_theta + R1(k, i, k, j) * lambda(k);
%             lambda_theta_theta = lambda_theta_theta + R1(k, i, k, j);
        end
        R3(i, j) = d2bt(:, i)' * d2bt(:, j) - lambda_theta_theta;
    end
end

R1 = R1 * dt;
R2 = R2 * dt;
R3 = R3 * dt;

% Compute the initial condition for eta
C1 = zeros(m + 1, m);

% Load the last residual field time step and estimate grad dot w_ss
if param.data_in_blocks.bool % if data are saved in several files
    name_file_U_temp = param.name_file_U_temp{end};
    load(name_file_U_temp, 'U');
else
    name_file_U_temp = param.name_file_U_temp;
    load(name_file_U_temp, 'U');
end
U = U(:, end, :);
U = reshape(U, [1, d, MX]);
del_U = gradient_mat(U, dX);
del_U = permute(del_U, [ndims(del_U) + 1, 1, ndims(del_U), 3 : ndims(del_U) - 1, 2]);
U = permute(U, [ndims(U) + 1, 1 : ndims(U)]);%(1,1,d,Mx ,My,(Mz))

% Compute the projection onto each topos
for q = 1 : m + 1
    % Small scale advected by large scale
    phi_q = phi_m_U(:, q, :);%(M,1,d)
    phi_q = permute(phi_q, [2, 3, 1]);%(1,d,M)
    phi_q = reshape(phi_q, [1, d, MX]);%(1,d,Mx,My,(Mz))
    dphi_q = gradient_mat(phi_q, dX);
    dphi_q = permute(dphi_q, [ndims(dphi_q) + 1, 1, ndims(dphi_q), 3 : ndims(dphi_q) - 1, 2]);
    %(1,1,d!,Mx,My,(Mz),d
    
    % compute the advection term_piq ::adv_piq
    adv_sl = bsxfun(@times, U, dphi_q); % 1 x (1) x d x Mx x My (x Mz) d
    clear dphi_q;
    adv_sl = sum(adv_sl, 3);
    adv_sl = permute(adv_sl, [1, 2, 4 : ndims(adv_sl), 3]);%(1 1 Mx My (Mz) d)
    
    % Large scale advected by small scale
    phi_q = permute(phi_q, [ndims(phi_q) + 1, 1 : ndims(phi_q)]);
    adv_ls = bsxfun(@times, phi_q, del_U);
    adv_ls = sum(adv_ls, 3);
    adv_ls = permute(adv_ls, [1, 2, 4 : ndims(adv_ls), 3]);%(1 1 Mx My (Mz) d)
    
    % Do the divergence free projection
    integ = adv_sl + adv_ls;
    integ = permute(integ, [3 : ndims(integ) - 1, 1, 2, ndims(integ)]); % [Mx, My, (Mz), 1, 1, d]
    integ = reshape(integ, [M, 1, d]);
    if param.eq_proj_div_free == 2
        if strcmp(param.type_data, 'turb2D_blocks_truncated')
            integ = integ - proj_div_propre(integ, MX, dX, true);
        else
            integ = integ - proj_div_propre(integ, MX, dX, false);
        end
    end
    integ = reshape(integ, [MX, 1, d]);
    integ = permute(integ, [ndims(integ) - 1, 1 : ndims(integ) - 2, ndims(integ)]);
    integ = reshape(integ, [1, 1, MX, d]);
    
    % Add the diffusion term if phi_0
    if q == m + 1
        U = reshape(U, [1, d, MX]);
        Lap_U = laplacian_mat(U, dX);
        Lap_U = nu * Lap_U;
        Lap_U = permute(Lap_U, [1, ndims(Lap_U) + 1, 2 : ndims(Lap_U)]);
        Lap_U = permute(Lap_U, [1, 2, 4 : ndims(Lap_U), 3]);%(1,1,Mx,My,(Mz),d)
        U = permute(U, [ndims(U) + 1, 1 : ndims(U)]);%(1,1,d,Mx ,My,(Mz))
        
        integ = integ - Lap_U;
    end
    
    % projection on phi_j
    for j = 1 : m
        phi_j = phi_m_U(:, j, :);
        phi_j = permute(phi_j, [4, 2, 1, 3]);%(1,1,M,d)
        phi_j = reshape(phi_j, [1, 1, MX, d]);%(1,1,Mx,My,(Mz),d)
        
        s_temp = integ .* phi_j; %(1,1,Mx,My,(Mz),d)
        clear phi_j;
        s_temp = sum(s_temp, ndims(s_temp));%(1,1,Mx,My,(Mz))
        
        % compute the integration of s_temp
        s_temp = integration_mat(s_temp, dX);
        C1(q,j) = - s_temp;
        
        clear s_temp;
    end
    clear adv_sl adv_ls Lap_del_pi;
end

end

function dU_del_dU = calculate_dU_del_dU(dU, d, MX, dX)
% Estimates (dw_ss * del) * dw_ss

% Reshape dU to be able to differentiate
dU = permute(dU, [3, 1, 2]);
dU = reshape(dU, [1, 1, d, 1, MX]);

if d == 2
    dx = diff_l(dU, 1, dX);
    dy = diff_l(dU, 2, dX);
    div_U = dx(1,1,1,1,:,:) + dy(1,1,2,1,:,:);
    div_U = permute(div_U, [5, 6, 1, 2, 3, 4]);
    dU = permute(dU, [5, 6, 1, 2, 3, 4]);
    dU_del_dU = bsxfun(@times, dU, div_U);
    dU_del_dU = sum(dU_del_dU, 3);
    dU_del_dU = permute(dU_del_dU, [1, 2, 4 : ndims(dU_del_dU), 3]); % [1, 1, Mx, My, d]
else
    dx = diff_l(dU, 1, dX);
    dy = diff_l(dU, 2, dX);
    dz = diff_l(dU, 3, dX);
    div_U = dx(1,1,1,1,:,:,:) + dy(1,1,2,1,:,:,:) + dz(1,1,3,1,:,:,:);
    div_U = permute(div_U, [5, 6, 7, 1, 2, 3, 4]);
    dU = permute(dU, [5, 6, 7, 1, 2, 3, 4]);
    dU_del_dU = bsxfun(@times, dU, div_U);
    dU_del_dU = sum(dU_del_dU, 4);
    dU_del_dU = permute(dU_del_dU, [1, 2, 3, 5 : ndims(dU_del_dU), 4]); % [1, 1, Mx, My, Mz, d]
end

end
