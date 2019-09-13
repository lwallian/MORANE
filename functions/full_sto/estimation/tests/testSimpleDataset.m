%% Test with real but simpler data
clear all, close all, clc;

type_data = 'incompact3D_noisy2D_40dt_subsampl_truncated';
is_in_blocks = false;
nb_modes = 2;
param = mock_param_structure(type_data, is_in_blocks, nb_modes);

% Estimate the POD
[param, bt] = POD_and_POD_knowing_phi(param);
dt = param.dt;

% Estimate eta for the non-orthogonal
param.folder_file_U_temp = fct_folder_temp(param);
param.name_file_U_temp = [param.folder_file_U_temp 'dsamp_' num2str(param.decor_by_subsampl.n_subsampl_decor) '_' '_U_temp'];
load(param.name_file_mode, 'phi_m_U')
load(param.name_file_U_temp, 'U');
phi = phi_m_U;
clear phi_m_U;

eta_estim = zeros(param.N_tot, nb_modes + 1, nb_modes);
for k = 1 : param.N_tot
    eta_estim(k, :, :) = operator_Q(U(:, k, :), phi, param);
end

% Estimate the noises in both cases
[result, pseudo_chol] = estimation_correlated_noises(param, bt);
% [result, pseudo_chol] = estimate_noise_non_orthogonal2(bt, eta_estim, nb_modes, param.N_tot, dt, 1, true, [param.lambda; 1.0]);
[result_est, pseudo_chol_est] = estimate_noise_non_orthogonal(bt, eta_estim, nb_modes, param.N_tot, dt, 1, true, [param.lambda; 1.0]);

err_result = (result - result_est).^2 / mean((result(:)).^2);
sq_mean_err_result = sqrt(mean(err_result(:)))

err_pseudo_chol = (pseudo_chol - pseudo_chol_est).^2 / mean((pseudo_chol(:)).^2);
sq_mean_err_pseudo_chol = sqrt(mean(err_pseudo_chol(:)))
