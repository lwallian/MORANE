function param = fct_name_file_diffusion_mode(param)
% Create the name of the file whe the diffusion mode(s) are saved
%

global correlated_model;
% Name of the saving file
if param.a_time_dependant
    dependance_on_time_of_a = '_a_time_dependant_';
else
    dependance_on_time_of_a = '_a_cst_';
end

if strcmp(param.decor_by_subsampl.choice_n_subsample, 'auto_shanon')
    str_threshold = num2str(param.decor_by_subsampl.spectrum_threshold);
    i_str_threshold = (str_threshold == '.');
    str_threshold(i_str_threshold)='_';
    %     str_threshold = num2str(param.decor_by_subsampl.spectrum_threshold);
    param.name_file_diffusion_mode =  ...
        [param.folder_data 'diffusion_mode_' param.type_data '_' ...
        num2str(param.nb_modes) '_modes' ...
        dependance_on_time_of_a ...
        'threshold_' str_threshold ];
else
    param.name_file_diffusion_mode =  ...
        [param.folder_data 'diffusion_mode_' param.type_data '_' ...
        num2str(param.nb_modes) '_modes' ...
        dependance_on_time_of_a ...
        'meth_' param.decor_by_subsampl.choice_n_subsample ];
end
if isfield(param,'N_estim')
    param.name_file_diffusion_mode = ...
        [ param.name_file_diffusion_mode ...
        '_p_estim_' num2str(param.period_estim)];
end
if correlated_model
    param.name_file_diffusion_mode = [param.name_file_diffusion_mode, 'correlated'];
end
param.name_file_diffusion_mode = ...
    [param.name_file_diffusion_mode '.mat'];