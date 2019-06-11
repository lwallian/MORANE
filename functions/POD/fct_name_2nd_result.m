function param = fct_name_2nd_result(param,modal_dt,reconstruction)
% Create the name of the file where the 2nd reuslts
%(the ROM defintion + ROM simmulations) are saved
%


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
            param.name_file_2nd_result=[ param.folder_results '2ndresult_' param.type_data '_' num2str(param.nb_modes) '_modes_' ...
                dependance_on_time_of_a char_filter ...
                '_decor_by_subsampl_' num2str(param.decor_by_subsampl.meth) ...
                '_choice_' num2str(param.decor_by_subsampl.choice_n_subsample)  ...
                '_thr_' num2str(param.decor_by_subsampl.spectrum_threshold) ...
                'fct_test_' param.decor_by_subsampl.test_fct ];
        case 'corr_time'
            param.name_file_2nd_result=[ param.folder_results '2ndresult_' param.type_data '_' num2str(param.nb_modes) '_modes_' ...
                dependance_on_time_of_a char_filter ...
                '_decor_by_subsampl_' num2str(param.decor_by_subsampl.meth) ...
                '_choice_' num2str(param.decor_by_subsampl.choice_n_subsample)  ...
                '_fct_test_' param.decor_by_subsampl.test_fct ];
    end
else
    param.name_file_2nd_result=[ param.folder_results '2ndresult_' param.type_data '_' num2str(param.nb_modes) '_modes_' ...
        dependance_on_time_of_a ];
end
param.name_file_2nd_result=[param.name_file_2nd_result '_fullsto'];
% if modal_dt
%     param.name_file_2nd_result=[param.name_file_2nd_result '_modal_dt'];
% end
if modal_dt == 1
    param.name_file_2nd_result=[param.name_file_2nd_result '_modal_dt'];
elseif modal_dt == 2
    param.name_file_2nd_result=[param.name_file_2nd_result '_real_dt'];
end
if ~ param.adv_corrected
    param.name_file_2nd_result=[param.name_file_2nd_result '_no_correct_drift'];    
end
if param.decor_by_subsampl.no_subampl_in_forecast
    param.name_file_2nd_result=[param.name_file_2nd_result '_no_subampl_in_forecast'];
end
if reconstruction
    param.reconstruction=true;
    param.name_file_2nd_result=[param.name_file_2nd_result '_reconstruction'];
else
    param.reconstruction=false;
end
param.name_file_2nd_result=[param.name_file_2nd_result '.mat'];
% save(param.name_file_2nd_result,'-v7.3');
% % save(param.name_file_2nd_result);
% clear C_deter C_sto L_deter L_sto I_deter I_sto
% % if param.big_data
%     toc;tic
%     disp('2nd result saved');
% % end


%%

% if param.decor_by_subsampl.bool
%     if strcmp(dependance_on_time_of_a,'a_t')
%         char_filter = [ '_on_' param.type_filter_a ];
%     else
%         char_filter = [];
%     end
%     param.name_file_1st_result=[ param.folder_results '1stresult_' param.type_data '_' num2str(param.nb_modes) '_modes_' ...
%         dependance_on_time_of_a char_filter ...
%         '_decor_by_subsampl_' num2str(param.decor_by_subsampl.meth) ...
%         '_choice_' num2str(param.decor_by_subsampl.choice_n_subsample)  ...
%         '_threshold_' num2str(param.decor_by_subsampl.spectrum_threshold)  ...
%         'fct_test_' param.decor_by_subsampl.test_fct ];
%     
% else
%     param.name_file_1st_result=[ param.folder_results '1stresult_' param.type_data '_' num2str(param.nb_modes) '_modes_' ...
%         dependance_on_time_of_a];
% end
% param.name_file_1st_result=[param.name_file_1st_result '_fullsto'];
% if ~ param.adv_corrected
%     param.name_file_1st_result=[param.name_file_1st_result '_no_correct_drift'];    
% end
% param.name_file_1st_result=[param.name_file_1st_result '.mat'];
% % save(param.name_file_1st_result);
% % clear coef_correctif_estim
% % if param.igrida
% %     toc;tic;
% %     disp('1st result saved');
% % end
% 
% 
% 
% % str_threshold = num2str(param.decor_by_subsampl.spectrum_threshold);
% % i_str_threshold = (str_threshold == '.');
% % str_threshold(i_str_threshold)='_';
% % %     str_threshold = num2str(param.decor_by_subsampl.spectrum_threshold);
% % param.name_file_noise_cov =  ...
% %     [param.folder_data 'noise_cov_' param.type_data '_' ...
% %     num2str(param.nb_modes) '_modes' ...
% %     'threshold_' str_threshold ];
% % if isfield(param,'N_estim')
% %     param.name_file_noise_cov = ...
% %         [ param.name_file_noise_cov ...
% %         '_p_estim_' num2str(param.period_estim)];
% % end
% % param.name_file_noise_cov = ...
% %     [param.name_file_noise_cov '.mat'];

