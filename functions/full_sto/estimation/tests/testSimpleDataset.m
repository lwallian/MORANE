%% Second debug of the correlated model. A simple dataset is taken, its POD
% is calculated and the estimations are carried out for the orthogonal and
% non orthogonal chronos cases

%% Test with real but simpler data
clear all, close all, clc;

type_data = 'inc3D_Re300_40dt_blocks_truncated';
% type_data = 'incompact3D_noisy2D_40dt_subsampl_truncated';
is_in_blocks = true;
nb_modes = 2;
param = mock_param_structure(type_data, is_in_blocks, nb_modes);

name_file_U_temp = param.name_file_U_temp;

% Estimate the POD
[param, bt] = POD_and_POD_knowing_phi(param);
dt = param.dt;

param.name_file_U_temp = name_file_U_temp;

% Estimate eta for the non-orthogonal
param.folder_file_U_temp = fct_folder_temp(param);
% param.name_file_U_temp = [param.folder_file_U_temp 'dsamp_' num2str(param.decor_by_subsampl.n_subsampl_decor) '_' '_U_temp'];
load(param.name_file_mode, 'phi_m_U')
% load(param.name_file_U_temp, 'U');
phi = phi_m_U;
clear phi_m_U;

%% Calcul of eta

% Initialization
eta_estim = zeros(param.N_tot, nb_modes + 1, nb_modes);
t_local = 1; % index of the snapshot in a file

if param.data_in_blocks.bool % if data are saved in several files  
    big_T = 1; % index of the file
    name_file_U_temp=param.name_file_U_temp{big_T}; % Name of the next file
    % Load new file
    load(name_file_U_temp, 'U');
else
    load(param.name_file_U_temp, 'U');
end

for t = 1 :  param.N_tot-1 % loop in time
% for t = 1 :  param.N_tot % loop in time
    if t_local > size(U, 2) % A new file needs to be loaded
        % initialization of the index of the snapshot in the file
        t_local = 1;
        % Incrementation of the file index
        big_T = big_T + 1;
        % Name of the new file
        name_file_U_temp = param.name_file_U_temp{big_T};
        % Load new file
        load(name_file_U_temp, 'U');
    end
    eta_estim(t, :, :) = operator_Q(U(:, t_local, :), phi, param);

    % Incrementation of the index of the snapshot in the file
    t_local = t_local + 1;
end

eta_estim(end, :, :)=nan;

% Estimate the noises in both cases
[result, pseudo_chol] = estimation_correlated_noises(param, bt);
[result_est, pseudo_chol_est] = estimate_noise_non_orthogonal(bt, eta_estim, nb_modes, param.N_tot, dt, 1, true, [param.lambda; 1.0]);
% [result_est, pseudo_chol_est] = estimate_noise_non_orthogonal(bt, eta_estim, nb_modes, param.N_tot, dt, 1, false);

err_result = (result - result_est).^2 / mean((result(:)).^2);
sq_mean_err_result = sqrt(mean(err_result(:)))

err_pseudo_chol = (pseudo_chol - pseudo_chol_est).^2 / mean((pseudo_chol(:)).^2);
sq_mean_err_pseudo_chol = sqrt(mean(err_pseudo_chol(:)))
