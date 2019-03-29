function [param,dphi_m_U] = grad_topos(type_data,nb_modes)
% Compute the gradient of topos
%
% m is the number of modes
% d ( = 2 or 3 ) is the dimension of the velocity field
% - phi ( M x m x d ) the first m spacial modes with their d coefficients in the M points of
% the grid
% - m_U ( M x 1 x d ) the time mean with its d coefficients in the M points of
% the grid
% - grid = [ x y (z)] where x, y, z are vectors, without
% repetitions, which define the grid. There are M point
%

%% Get Param
param.folder_results = [ pwd '/resultats/current_results/'];
current_pwd = pwd; cd ..
param.folder_data = [ pwd '/data/' ];
cd(current_pwd); clear current_pwd
param.type_data = type_data;

trun='_truncated';
if (length(param.type_data)>= 10) ...
        && strcmp(param.type_data(end-9:end),trun)
    param.data_in_blocks.bool = true;
    param.data_in_blocks.type_whole_data = param.type_data(1:(end-10));
    param.type_data_temp = [ param.data_in_blocks.type_whole_data '1'];
    param=read_param_data(param.type_data_temp,param.folder_data, ...
        param.data_in_blocks.type_whole_data);
%     param=read_param_data(param.type_data,param.folder_data, ...
%         param.data_in_blocks.type_whole_data);
else
    param.data_in_blocks.bool = false;
    param.data_in_blocks.type_whole_data = nan;
    param=read_param_data(param.type_data,param.folder_data);
end
% keyboard;

%% Pre-treatement

% Spatial modes
param.type_data = type_data; clear type_data
param.nb_modes = nb_modes; clear nb_modes
param.name_file_mode = [ param.folder_data ...
    'mode_' param.type_data '_' num2str(param.nb_modes) '_modes.mat'];
% file_phi=param.name_file_mode;
load(param.name_file_mode,'phi_m_U');

% Get size
[~,m,d]=size(phi_m_U);
m=m-1;

% Get data about the grid
dX=param.dX; % Space steps
MX=param.MX; % Numbers of space steps in each directions

% Coding trick
idx='';
for k_dim=1:d
    idx = [idx ',:'];
end

%% Reshape spatial modes
% The treatement of time mean of U (m_U) is done with the treatement of the modes phi
% The form of the arrays should be adapted to the form of the grid to make
% the derivations easier
phi_m_U = permute(phi_m_U,[2 3 1]); % (m+1) x d x M
% (m+1)(number of modes+1) x d(number of velocity components) x M(space)
phi_m_U = reshape(phi_m_U, [m+1 d MX]); % (m+1) x d x Mx x My (x Mz)

% Gradient of the modes phi % (m+1) x d x Mx x My (x Mz) x d
dphi_m_U = gradient_mat(phi_m_U,dX);
    
%% Save
param_from_file = param;
param.name_file_grad_mode = [ param.folder_data....
    'grad_mode_' param.type_data '_' num2str(param.nb_modes) '_modes.mat'];
save(param.name_file_grad_mode,'param_from_file','dphi_m_U','-v7.3');

