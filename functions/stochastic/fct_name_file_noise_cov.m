function param = fct_name_file_noise_cov(param)
% Create the name of the file whe the diffusion mode(s) are saved
%

global estim_rmv_fv;
global correlated_model;
if strcmp(param.decor_by_subsampl.choice_n_subsample, 'auto_shanon')
    str_threshold = num2str(param.decor_by_subsampl.spectrum_threshold);
    i_str_threshold = (str_threshold == '.');
    str_threshold(i_str_threshold)='_';
    %     str_threshold = num2str(param.decor_by_subsampl.spectrum_threshold);
    param.name_file_noise_cov =  ...
        [param.folder_data 'noise_cov_' param.type_data '_' ...
        num2str(param.nb_modes) '_modes' ...
        'threshold_' str_threshold ];
else
    param.name_file_noise_cov =  ...
        [param.folder_data 'noise_cov_' param.type_data '_' ...
        num2str(param.nb_modes) '_modes' ...
        'meth_' param.decor_by_subsampl.choice_n_subsample ];
end
if correlated_model
    param.name_file_noise_cov = [param.name_file_noise_cov, 'correlated'];
end
if estim_rmv_fv
    param.name_file_noise_cov = [param.name_file_noise_cov '_estim_rmv_fv'];
end
if param.eq_proj_div_free == 2
    param.name_file_noise_cov = [param.name_file_noise_cov '_DFSPN'];    
end
param.name_file_noise_cov = ...
    [param.name_file_noise_cov '.mat'];

