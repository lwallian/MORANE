clear all;
close all;

%% Parameters of the script
load('F:\GitHub\PODFS\resultats\current_results\2ndresult_incompact3d_wake_episode3_cut_truncated_4_modes__a_cst__decor_by_subsampl_bt_decor_choice_auto_shanon_threshold_0.0005fct_test_b_fullsto.mat');
clearvars -except param
param.name_file_diffusion_mode = 'F:\GitHub/data/test_z.mat';
param.name_file_mode = 'F:\GitHub/data/test_phi_m_U.mat';
param.dX = (1/3) * param.dX;
param.dX(2) = 2 * param.dX(2)
dX = param.dX;
param.MX = 3*param.MX;
MX = param.MX;
param.M = 9*param.M;
M = param.M;
d = param.d;

%% Creating fake z and phi_m_U
% using z as an identity matrix
% load(param.name_file_diffusion_mode);
% creating a new z
[X, Y] = meshgrid(0:dX(1):dX(1)*(MX(1)-1),0:dX(2):dX(2)*(MX(2)-1));
Z = exp(-(((X - 0.5*dX(1)*MX(1))./(25*dX(1))).^2+((Y - 0.5*dX(2)*MX(2))./(25*dX(2))).^2))'; % Mx x My

z = zeros([MX 1 d d]);
z(:,:,1,1,1) = 0.4*Z;
z(:,:,1,1,2) = 0.1*Z;
z(:,:,1,2,1) = 0.2*Z;
z(:,:,1,2,2) = 0.3*Z;
z = reshape(z, [M 1 d d]);
save(param.name_file_diffusion_mode,'z');
% using phi_m_U of incompact3d_wake_episode3_cut 
%load(param.name_file_mode);
%creating a new "random-ish" phi_m_U
phi_m_U = rand([MX param.nb_modes+1 d]); % Mx x My x m+1 x d
for i=1:param.nb_modes+1
    phi_m_U(:,:,i,1) = 0.1*i*Z;
    phi_m_U(:,:,i,2) = 0.2*i*Z;
end
phi_m_U = reshape(phi_m_U, [M param.nb_modes+1 d]);
save(param.name_file_mode,'phi_m_U');
%% Computing L_sto with the first method
[I_sto,L_sto,C_sto] = param_ODE_bt_sto(param.name_file_mode, param, param.grid);
L_sto

%% Computing L_sto with the second method
F = coefficients_sto_bis(param)