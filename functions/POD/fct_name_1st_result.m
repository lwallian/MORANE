function param = fct_name_1st_result(param)
% Create the name of the file where the 1st reuslts (the ROM defintion) are saved
%
global stochastic_integration;
global estim_rmv_fv;
global correlated_model;

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
            param.name_file_1st_result=[ param.folder_results '1stresult_' param.type_data '_' num2str(param.nb_modes) '_modes_' ...
                dependance_on_time_of_a char_filter ...
                '_decor_by_subsampl_' num2str(param.decor_by_subsampl.meth) ...
                '_choice_' num2str(param.decor_by_subsampl.choice_n_subsample)  ...
                '_thr_' num2str(param.decor_by_subsampl.spectrum_threshold)  ...
                'fct_test_' param.decor_by_subsampl.test_fct ];
        otherwise
            param.name_file_1st_result=[ param.folder_results '1stresult_' param.type_data '_' num2str(param.nb_modes) '_modes_' ...
                dependance_on_time_of_a char_filter ...
                '_decor_by_subsampl_' num2str(param.decor_by_subsampl.meth) ...
                '_choice_' num2str(param.decor_by_subsampl.choice_n_subsample)  ...
                'fct_test_' param.decor_by_subsampl.test_fct ];
    end
else
    param.name_file_1st_result=[ param.folder_results '1stresult_' param.type_data '_' num2str(param.nb_modes) '_modes_' ...
        dependance_on_time_of_a];
end
param.name_file_1st_result=[param.name_file_1st_result '_fullsto'];
if ~ param.adv_corrected
    param.name_file_1st_result=[param.name_file_1st_result '_no_correct_drift'];    
end
if correlated_model
    param.name_file_1st_result=[param.name_file_1st_result '_correlated'];
end
param.name_file_1st_result=[param.name_file_1st_result '_integ_' stochastic_integration];
if estim_rmv_fv
    param.name_file_1st_result=[param.name_file_1st_result '_estim_rmv_fv'];
    param.estim_rmv_fv = true;
end
param.name_file_1st_result=[param.name_file_1st_result '.mat'];
% save(param.name_file_1st_result);
% clear coef_correctif_estim
% if param.igrida
%     toc;tic;
%     disp('1st result saved');
% end



% str_threshold = num2str(param.decor_by_subsampl.spectrum_threshold);
% i_str_threshold = (str_threshold == '.');
% str_threshold(i_str_threshold)='_';
% %     str_threshold = num2str(param.decor_by_subsampl.spectrum_threshold);
% param.name_file_noise_cov =  ...
%     [param.folder_data 'noise_cov_' param.type_data '_' ...
%     num2str(param.nb_modes) '_modes' ...
%     'threshold_' str_threshold ];
% if isfield(param,'N_estim')
%     param.name_file_noise_cov = ...
%         [ param.name_file_noise_cov ...
%         '_p_estim_' num2str(param.period_estim)];
% end
% param.name_file_noise_cov = ...
%     [param.name_file_noise_cov '.mat'];

