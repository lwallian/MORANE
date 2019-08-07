function param = fct_name_file_noise_cov(param)
% Create the name of the file whe the diffusion mode(s) are saved
%
global estim_rmv_fv;

str_threshold = num2str(param.decor_by_subsampl.spectrum_threshold);
i_str_threshold = (str_threshold == '.');
str_threshold(i_str_threshold)='_';
%     str_threshold = num2str(param.decor_by_subsampl.spectrum_threshold);
param.name_file_noise_cov =  ...
    [param.folder_data 'noise_cov_' param.type_data '_' ...
    num2str(param.nb_modes) '_modes' ...
    'threshold_' str_threshold ];
if isfield(param,'N_estim')
    param.name_file_noise_cov = ...
        [ param.name_file_noise_cov ...
        '_p_estim_' num2str(param.period_estim)];
end
if estim_rmv_fv
    param.name_file_noise_cov = [param.name_file_noise_cov '_estim_rmv_fv'];
end
param.name_file_noise_cov = ...
    [param.name_file_noise_cov '.mat'];

