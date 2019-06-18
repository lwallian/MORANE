function file_save = fct_file_save_1st_result(param)
% Create the file name to save the first results of the POD
%
global stochastic_integration;

if param.a_time_dependant
    dependance_on_time_of_a = '_a_time_dependant_';
else
    dependance_on_time_of_a = '_a_cst_';
end
if param.decor_by_subsampl.bool
    if strcmp(dependance_on_time_of_a,'a_t')
        char_filter = [ '_on_' param.type_filter_a ];
    else
        char_filter = [];
    end
    switch param.decor_by_subsampl.choice_n_subsample
        case 'auto_shanon'
            file_save=[ param.folder_results '1stresult_' param.type_data '_' num2str(param.nb_modes) '_modes_' ...
                dependance_on_time_of_a char_filter ...
                '_decor_by_subsampl_' num2str(param.decor_by_subsampl.meth) ...
                '_choice_' num2str(param.decor_by_subsampl.choice_n_subsample)  ...
                '_threshold_' num2str(param.decor_by_subsampl.spectrum_threshold)  ...
                'fct_test_' param.decor_by_subsampl.test_fct ];
        case 'corr_time'
            file_save=[ param.folder_results '1stresult_' param.type_data '_' num2str(param.nb_modes) '_modes_' ...
                dependance_on_time_of_a char_filter ...
                '_decor_by_subsampl_' num2str(param.decor_by_subsampl.meth) ...
                '_choice_' num2str(param.decor_by_subsampl.choice_n_subsample)  ...
                'fct_test_' param.decor_by_subsampl.test_fct ];
    end
else
    file_save=[ param.folder_results '1stresult_' param.type_data '_' num2str(param.nb_modes) '_modes_' ...
        dependance_on_time_of_a ];
end
if isfield(param,'N_estim')
    file_save=[file_save '_p_estim_' num2str(param.period_estim)];
end
file_save = [file_save '_integ_' stochastic_integration];
file_save=[file_save '.mat'];