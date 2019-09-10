%% Test with real but simpler data
clear all, close all, clc;

type_data = 'incompact3D_noisy2D_40dt_subsampl_truncated';
is_in_blocks = false;
nb_modes = 2;
param = mock_param_structure(type_data, is_in_blocks, nb_modes);

[param, bt] = POD_and_POD_knowing_phi(param);
[R1, R2, R3] = fct_comp_correlated_RHS(param, bt);

theta_theta = R1;
    
% Compute Mi_sigma
Mi_sigma = R2 ./ param.N_tot;

% Compute xi_xi_inf
xi_xi_inf = R3;

clear R1 R2 R3;

theta_theta = reshape(theta_theta, [nb_modes * (nb_modes + 1), nb_modes * (nb_modes + 1)]);
xi_xi_inf = reshape(xi_xi_inf, [nb_modes, nb_modes]);
theta_xi = zeros(nb_modes * (nb_modes + 1), nb_modes);

result1 = [xi_xi_inf; theta_xi];
result2 = [theta_xi'; theta_theta];
result = [result1, result2];
clear result1 result2;

% Force the symetry and the positivity of the matrix
result = 1 / 2 * (result + result');
[V, D] = eig(result);
D = diag(D);
D(D < 0) = 0;
result = V * diag(D) * V';

% Now doing the same but with the non-orthogonal formulas
% Start by calculating beta
param.lambda(nb_modes + 1) = 1;
theta_theta = reshape(theta_theta, [nb_modes + 1, nb_modes, nb_modes + 1, nb_modes]);
beta = zeros(nb_modes + 1, nb_modes, nb_modes + 1, nb_modes);
for p = 1 : nb_modes + 1
    beta(p, :, :, :) = theta_theta(p, :, :, :) ./ param.lambda(p);
end

% Load the topos and the residual of the field (no big data...)
param.folder_file_U_temp = fct_folder_temp(param);
param.name_file_U_temp = [param.folder_file_U_temp 'dsamp_' num2str(param.decor_by_subsampl.n_subsampl_decor) '_' '_U_temp'];
load(param.name_file_mode, 'phi_m_U')
load(param.name_file_U_temp, 'U');
phi = phi_m_U;
clear phi_m_U;

% Estimate the eta from the field
eta_estim = zeros(param.N_tot, nb_modes + 1, nb_modes);
for k = 1 : param.N_tot
    eta_estim(k, :, :) = operator_Q(U(:, k, :), phi, param);
end

% Estimate psi
bt_x = cat(2, bt, ones(param.N_tot, 1));
psi = zeros(param.M, nb_modes + 1, param.d);
for t = 1 : param.N_tot
    for k = 1 : param.d
        for p = 1 : nb_modes + 1
            psi(:, p, k) = psi(:, p, k) + bt_x(t, p) .* U(:, t, k);
        end
    end
end
psi = psi ./ param.N_tot;

% Estimate gamma
gamma = zeros(nb_modes + 1, nb_modes, nb_modes + 1, nb_modes);
dU = diff(U, 1, 2);
% for k = 1 : param.N_tot - 1
for k = 1 : param.N_tot - 1
    R_pi = operator_R(dU(:, k, :), psi, param);
    Q_qj = operator_Q(dU(:, k, :), phi, param);
    for p = 1 : nb_modes + 1
        for i = 1 : nb_modes
            for q = 1 : nb_modes + 1
                for j = 1 : nb_modes
                    gamma(p, i, q, j) = gamma(p, i, q, j) + ...
                        R_pi(p, i) * Q_qj(q, j);
                end
            end
        end
    end
end
gamma = gamma .* param.dt ./ (param.N_tot - 1);

% Estimate theta_theta for the non-orthogonal case
G_pq = bt_x' * bt_x ./ param.N_tot; % ou T ?
theta_theta_estim = zeros(nb_modes + 1, nb_modes, nb_modes + 1, nb_modes);
for i = 1 : nb_modes
    for q = 1 : nb_modes + 1
        for j = 1 : nb_modes
            theta_theta_estim(:, i, q, j) = linsolve(G_pq, beta(:, i, q, j) - gamma(:, i, q, j));
        end
    end
end

% Estimate xi_xi for the non-orthogonal case
d2bt = diff(bt, 2, 1);
xi_xi_estim = zeros(nb_modes, nb_modes);
for i = 1 : nb_modes
    for j = 1 : nb_modes
        kappa = 0;
        for p = 1 : nb_modes + 1
            for q = 1 : nb_modes + 1
                kappa = kappa + theta_theta(p, i, q, j);
%                 kappa = kappa + theta_theta_estim(p, i, q, j);
            end
        end
        xi_xi_estim(i, j) = d2bt(:, i)' * d2bt(:, j) - kappa;
    end
end
