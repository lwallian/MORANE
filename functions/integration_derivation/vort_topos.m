function vort_topos(param,dphi_m_U)
% Compute topos vorticity and rate-of-strain tensors
%

% Load
load(param.name_file_grad_mode,'param_from_file','dphi_m_U');

% Topos vorticity tensors
if param.d == 2
    omega_phi_m_U = dphi_m_U(:,2,:,:,1) - dphi_m_U(:,1,:,:,2);
elseif param.d == 3
    z_ref = floor(param.MX(3)/2);
    omega_phi_m_U = dphi_m_U(:,2,:,:,z_ref,1) - dphi_m_U(:,1,:,:,z_ref,2);
else
    error('The dimension should be 2 or 3');
end
omega_phi_m_U = (1/2) * omega_phi_m_U;
omega_phi_m_U = reshape(omega_phi_m_U, [(param.nb_modes+1) 1 prod(param.MX(1:2)) 1]);
omega_phi_m_U = permute(omega_phi_m_U, [3 1 2 4]);
    
%% Save
param_from_file = param;
param.name_file_omega_mode = [ param.folder_data ...
    '2dvort_mode_' param.type_data '_' num2str(param.nb_modes) '_modes.mat'];
save(param.name_file_omega_mode,'param_from_file',...
    'omega_phi_m_U','-v7.3');

