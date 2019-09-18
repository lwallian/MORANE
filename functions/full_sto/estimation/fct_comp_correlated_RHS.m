function [R1, R2, C1, C2] = fct_comp_correlated_RHS(param, bt)
%FCT_COMP_CORRELATED_RHS Estimates the noise statistics in the correlated
%non resolved modes scheme given the PCA residual and the chronos functions
%   @param param: structure with lots of parameters concerning the current
%   simulation
%   @param bt: chronos function of the current model
%   @return R1: theta_theta term
%   @return R2: xi_xi_inf term
%   @return C1: initial value for Mi_ss
%   @return C2: initial value for eta
%
% Author: Agustin PICARD, intern @ Scalian with Valentin RESSEGUIER as
% supervisor
%

dt = param.dt;
N_tot = param.N_tot;
m = param.nb_modes;
M = param.M;
dX = param.dX;
MX = param.MX;
d = param.d;
lambda = param.lambda;

% The last two time steps are not used
N_tot = N_tot - 2;

% Estimate the second wrt time of the chronos
diff_bt = (bt(2 : end, :, :) - bt(1 : end - 1, :, :)) / (dt);
diff_bt(end, :, :, :) = [];
bt(1, :, :) = [];
bt(end, :, :) = [];
d2bt = (diff_bt(2 : end, :, :) - diff_bt(1 : end - 1, :, :)) / (dt);
d2bt_c = d2bt - mean(d2bt, 1);

%% load U

% Initialization
t_local = 1; % index of the snapshot in a file
if param.data_in_blocks.bool % if data are saved in several files
    
    big_T = param.data_in_blocks.nb_blocks;
    name_file_U_temp=param.name_file_U_temp{big_T}; % Name of the next file
    % Load new file
    load(name_file_U_temp, 'U');
    U_last = U(:, end-1, :); clear U;    
    
    big_T = 1; % index of the file
    name_file_U_temp=param.name_file_U_temp{big_T}; % Name of the next file
    % Load new file
    load(name_file_U_temp, 'U');
    
    U_first = U(:,2, :);
    mean_dU = (1/(N_tot*dt)) * ( U_last - U_first) ;
    
    len_local = size(U, 2)-2;
    % Differentiate it wrt time
    U(:, 1, :) = [];
    
    dU = (U(:, 2 : end, :) - U(:, 1 : end - 1, :)) ./ dt;
    dU = dU - mean_dU;
else
    name_file_U_temp=param.name_file_U_temp; % Name of the next file
    load(name_file_U_temp, 'U');
    U(:, 1, :) = [];
    U(:, end, :) = [];
    dU = (U(:, 2 : end, :) - U(:, 1 : end - 1, :)) ./ dt;
    dU = dU - mean(dU, 2);
end

U_last = U(:,end, :);
clear U

%compute the sum(b_p * d2bt_i * dX_res)
del_pi = zeros(M, m + 1, m, d); % (M,m(p),m(i),d)
% add the constant term to bt
bt = cat(2, bt, ones(N_tot, 1));
lambda(m + 1) = 1;

if iscell(param.name_file_U_temp)
    N_final = 2;
else
    N_final = 1;
end

for t = 1 : N_tot - N_final % loop in time
    if iscell(param.name_file_U_temp) && ...
            (t_local > len_local) % A new file needs to be loaded
        % initialization of the index of the snapshot in the file
        t_local = 1;
        % Incrementation of the file index
        big_T = big_T + 1;
        % Name of the new file
        name_file_U_temp = param.name_file_U_temp{big_T};
        % Load new file
        load(name_file_U_temp, 'U');
        
        % Do the pertinent manipulations to the current snapshot
        len_local = size(U, 2);
        U = cat(2, U_last, U);
        dU = (U(:, 2 : end, :) - U(:, 1 : end - 1, :)) ./ dt;
        U_last = U(:,end, :);
        clear U
        dU = dU - mean_dU;
    end
    for k = 1 : d
        for i = 1 : m
            for p = 1 : m + 1
                del_pi(:, p, i, k) = del_pi(:, p, i, k) ...
                    + dU(:, t_local, k) * d2bt_c(t, i) ...
                    * bt(t, p) / lambda(p);
            end
        end
    end
    % Incrementation of the index of the snapshot in the file
    t_local = t_local + 1;
end

% C2 (the initial conditions for Mi_ss) is calculated as the projection
% onto the topos for (d_wss del) d_wss for the last time step available in
% the dataset
C2_mat = calculate_dU_del_dU(dU(:, end, :), d, MX, dX);
clear dU;

% Do the divergence free projection before projecting onto each topos
C2_mat = reshape(C2_mat, [M, 1, d]);
if param.eq_proj_div_free == 2
    if strcmp(param.type_data, 'turb2D_blocks_truncated')
        C2_mat = C2_mat - proj_div_propre(C2_mat, MX, dX, true);
    else
        C2_mat = C2_mat - proj_div_propre(C2_mat, MX, dX, false);
    end
end

% Load the topos
load(param.name_file_mode, 'phi_m_U')

% Compute the projection over each mode for Mi_sigma
C2 = zeros(m, 1);
C2_mat = reshape(C2_mat, [1, 1, MX, d]);
for j = 1 : m
    phi_j = phi_m_U(:, j, :);
    phi_j = permute(phi_j, [4, 2, 1, 3]);%(1,1,M,d)
    phi_j = reshape(phi_j, [1, 1, MX, d]);%(1,1,Mx,My,(Mz),d)
    
    c_temp = C2_mat .* phi_j;
    clear phi_j;
    c_temp = sum(c_temp, ndims(c_temp));
    
    % compute the integration of c_temp
    c_temp = integration_mat(c_temp, dX);
    C2(j) = -c_temp;
    clear c_temp;
end

% Compute theta_theta:
R1 = zeros(m + 1, m, m + 1, m);
for p = 1 : m + 1
    for i = 1 : m
        R1(p, i, :, :) = operator_Q(del_pi(:, p, i, :), phi_m_U, param);
    end
end
clear del_pi;

% Compute xi_xi_inf
R2 = zeros(m, m);
R1 = R1 * dt / N_tot;
for i = 1 : m
    for j = 1 : m
        lambda_theta_theta = 0;
        for k = 1 : m + 1
            lambda_theta_theta = lambda_theta_theta + R1(k, i, k, j) * lambda(k);
        end
        R2(i, j) = d2bt_c(:, i)' * d2bt_c(:, j) * dt / N_tot - lambda_theta_theta;
    end
end

% Compute the initial condition for eta
% Load the last residual field time step and estimate grad dot w_ss
if param.data_in_blocks.bool % if data are saved in several files
    name_file_U_temp = param.name_file_U_temp{end};
    load(name_file_U_temp, 'U');
else
    name_file_U_temp = param.name_file_U_temp;
    load(name_file_U_temp, 'U');
end

% Just like before, C1 (the initial conditions for eta) is estimated as the
% operator Q applied to the last available datapoints in the training set.
U = U(:, end, :);
C1 = operator_Q(U, phi_m_U, param);

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
