function [U,param]=pre_treatement_of_U(param_ref)
% Remove global time average value to a specific files
%

% Load the file
if isfield(param_ref.data_in_blocks,'type_whole_data') % if data are saved in several files
    [U,param]=read_data(param_ref.type_data,param_ref.folder_data, ...
        param_ref.data_in_blocks.type_whole_data);
else
    [U,param]=read_data(param_ref.type_data,param_ref.folder_data);
end
% M is the number of points in the spatial grid
% N is the number of snapshots on the left file U1
% d = 2 or 3 is the dimension of the space
[param.M , param.N_tot, param.d] = size(U);
% Gather the structures param_ref and param
param_ref=rmfield(param_ref,'type_data');
param_ref=rmfield(param_ref,'folder_data');
param = mergestruct(param,param_ref);
% Path for the file which will contain residual velocity of U
% param.name_file_U_temp=[param.folder_data param.type_data '_U_temp'];
param.name_file_U_centered=[param.folder_data param.type_data '_U_centered'];

%% Remove average value
% Load average value
if isfield(param.data_in_blocks,'type_whole_data') % if data are saved in several files
    type_data=[param.data_in_blocks.type_whole_data num2str(0)];
else
    type_data=[param.type_data num2str(0)];    
end
param.name_file_mU=[param.folder_data type_data '_U_centered'];
load(param.name_file_mU);
% load(param.name_file_mode);
% Remove average value
U= bsxfun(@minus,U , m_U );
clear m_U

%% Save
% % save(param.name_file_U_temp,'U','-v7.3');
% save(param.name_file_U_centered,'U','-v7.3');
if isfield(param_ref.data_in_blocks,'type_whole_data') % if data are saved in several files
    save(param.name_file_U_centered,'U')
else
    save(param.name_file_U_centered,'U','-v7.3')
end
end