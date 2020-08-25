function param = fct_name_1st_result_new(param)
% Create the name of the file where the 1st reuslts (the ROM defintion) are saved
%
global stochastic_integration;
global estim_rmv_fv;
% global choice_n_subsample;
global correlated_model;

if param.decor_by_subsampl.bool
%     switch choice_n_subsample
    switch param.decor_by_subsampl.choice_n_subsample
        case 'auto_shanon'
            param.name_file_1st_result=[ param.folder_results '1stresult_' ...
                param.type_data '_' num2str(param.nb_modes) '_modes_' ...
                param.decor_by_subsampl.choice_n_subsample  ...
                '_threshold_' num2str(param.decor_by_subsampl.spectrum_threshold)  ...
                param.decor_by_subsampl.test_fct ];
%         case {'lms', 'truncated', 'htgen', 'corr_time'} % corr_time for compatibility
        otherwise
            param.name_file_1st_result=[ param.folder_results '1stresult_' ...
                param.type_data '_' num2str(param.nb_modes) '_modes_' ...
                param.decor_by_subsampl.choice_n_subsample  ...
                param.decor_by_subsampl.test_fct ];
    end
else
    param.name_file_1st_result=[ param.folder_results '1stresult_' ...
        param.type_data '_' num2str(param.nb_modes) '_modes_' ...
        ];
end
param.name_file_1st_result=[param.name_file_1st_result '_fullsto'];
mkdir(param.name_file_1st_result)
param.name_file_1st_result=[param.name_file_1st_result '/'];
if ~correlated_model
    if ~ param.adv_corrected
        param.name_file_1st_result=[param.name_file_1st_result '_no_correct_drift'];
    end
end
param.name_file_1st_result=[param.name_file_1st_result '_integ_' stochastic_integration];
if estim_rmv_fv
    param.name_file_1st_result=[param.name_file_1st_result '_estim_rmv_fv'];
    param.estim_rmv_fv = true;
end
if correlated_model
    param.name_file_1st_result=[param.name_file_1st_result '_correlated'];
end
if ~ param.decor_by_subsampl.bug_sampling
    param.name_file_1st_result = [param.name_file_1st_result, '_noBugSubsampl'];
end
if param.eq_proj_div_free == 2
    param.name_file_1st_result = [param.name_file_1st_result '_DFSPN'];    
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

