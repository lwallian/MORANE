function tensor_topos(param,dphi_m_U)
% Compute topos vorticity and rate-of-strain tensors
%

%% Load
load(param.name_file_grad_mode,'param_from_file','dphi_m_U');

%% Topos vorticity tensors
if param.d == 2
    Omega_phi_m_U = dphi_m_U - permute( dphi_m_U , [1 5 3 4 2]);
elseif param.d == 3
    Omega_phi_m_U = dphi_m_U - permute( dphi_m_U , [1 6 3 4 5 2]);
else
    error('The dimension should be 2 or 3');
end
clear dphi_m_U
Omega_phi_m_U = (1/2) * Omega_phi_m_U;
Omega_phi_m_U = reshape(Omega_phi_m_U, [(param.nb_modes+1) param.d param.M param.d]);
Omega_phi_m_U = permute(Omega_phi_m_U, [3 1 2 4]);
% Omega_phi_m_U = reshape(Omega_phi_m_U,...
%     [param.M (param.nb_modes+1) param.d param.d]);

%% Save
param_from_file = param;
param.name_file_tensor_mode = [ param.folder_data ...
    'tensor_mode_' param.type_data '_' num2str(param.nb_modes) '_modes.mat'];
save(param.name_file_tensor_mode,'param_from_file',...
    'Omega_phi_m_U','-v7.3');
clear Omega_phi_m_U

%% Load
load(param.name_file_grad_mode,'param_from_file','dphi_m_U');

%% Topos rate-of-strain tensors
if param.d == 2
    S_phi_m_U = dphi_m_U + permute( dphi_m_U , [1 5 3 4 2]);
elseif param.d == 3
    S_phi_m_U = dphi_m_U + permute( dphi_m_U , [1 6 3 4 5 2]);
else
    error('The dimension should be 2 or 3');
end
clear dphi_m_U
S_phi_m_U = (1/2) * S_phi_m_U;
S_phi_m_U = reshape(S_phi_m_U, [(param.nb_modes+1) param.d param.M param.d]);
S_phi_m_U = permute(S_phi_m_U, [3 1 2 4]);
    
%% Save
param_from_file = param;
param.name_file_tensor_mode = [ param.folder_data ...
    'tensor_mode_' param.type_data '_' num2str(param.nb_modes) '_modes.mat'];
save(param.name_file_tensor_mode,'param_from_file',...
    'S_phi_m_U','-v7.3','-append');
clear S_phi_m_U

