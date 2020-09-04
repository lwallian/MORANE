function folder_file_U_temp = fct_folder_temp(param)
% Create the file name to save the first results of the POD
%
global stochastic_integration;
threshold_str = num2str(param.decor_by_subsampl.spectrum_threshold);
iii = (threshold_str=='.');
threshold_str(iii)='_';

if param.decor_by_subsampl.bool
    folder_file_U_temp = ...
        [ param.folder_data 'folder_file_temp_' param.type_data '_' ...
        num2str(param.nb_modes) '_modes_' ...
        '_decor_by_subsampl_' num2str(param.decor_by_subsampl.meth) ...
        '_choice_' num2str(param.decor_by_subsampl.choice_n_subsample)  ...
        '_threshold_' threshold_str  ...
        'fct_test_' param.decor_by_subsampl.test_fct ];
    
else
    folder_file_U_temp = ...
        [ param.folder_data '/folder_file_temp_' param.type_data '_' num2str(param.nb_modes) '_modes_' ...
        dependance_on_time_of_a ];
end
if isfield(param,'N_estim')
    folder_file_U_temp=[folder_file_U_temp '_p_estim_' num2str(param.period_estim)];
end
if strcmp(stochastic_integration, 'Ito')
    folder_file_U_temp = [ folder_file_U_temp, '_Ito'];  
elseif strcmp(stochastic_integration, 'Str')
    folder_file_U_temp = [ folder_file_U_temp, '_Str'];  
end
if param.noise_type > 0
    folder_file_U_temp = [ folder_file_U_temp, ...
        '_noise_type_' num2str(param.noise_type)];    
end

mkdir(folder_file_U_temp);

folder_file_U_temp=[folder_file_U_temp '/'];
