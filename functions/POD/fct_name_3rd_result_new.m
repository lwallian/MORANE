function [param,param_obs] = fct_name_3rd_result_new(param,param_obs,modal_dt)
% Create the name of the file where the 2nd reuslts
%(the ROM defintion + ROM simmulations) are saved
%
global stochastic_integration;
global estim_rmv_fv;
global choice_n_subsample;
global correlated_model;

param.name_file_3rd_result = [ param.folder_results ...
    param.type_data '_' num2str(param.nb_modes) '_modes_'  ...
    choice_n_subsample ];
if strcmp(choice_n_subsample, 'auto_shanon')
    param.name_file_3rd_result = [param.name_file_3rd_result  '_threshold_' ...
        num2str(param.decor_by_subsampl.spectrum_threshold)];
end
param.name_file_3rd_result = [param.name_file_3rd_result  param.decor_by_subsampl.test_fct  '/'];
if ~correlated_model
    if strcmp(choice_n_subsample, 'auto_shanon') && modal_dt
        param.name_file_3rd_result = [param.name_file_3rd_result  '_modal_dt'];
    end
    if ~ param.adv_corrected
        param.name_file_3rd_result = [param.name_file_3rd_result  '_no_correct_drift'];
    end
end


param.name_file_3rd_result = [param.name_file_3rd_result '_integ_' stochastic_integration];
if correlated_model
    param.name_file_3rd_result = [param.name_file_3rd_result '_correlated'];
end
if estim_rmv_fv
    param.name_file_3rd_result=[param.name_file_3rd_result '_estim_rmv_fv'];
end
switch param.svd_pchol
    case 1
        param.name_file_3rd_result=[param.name_file_3rd_result '_svd_pchol'];
    case 2
        param.name_file_3rd_result=[param.name_file_3rd_result '_svd_pchol2'];
end
if param.eq_proj_div_free == 2
    param.name_file_3rd_result = [param.name_file_3rd_result '_DFSPN'];
end
%         param.name_file_3rd_result = [param.name_file_3rd_result  '_integ_Ito'];
% if param.svd_pchol
%     param.name_file_3rd_result = [param.name_file_3rd_result  '_svd_pchol'];
% end
% param.name_file_3rd_result = [param.name_file_3rd_result  '/'  param_obs.assimilate  ...
%     '/_DADuration_'  num2str(param_obs.SECONDS_OF_SIMU)  '_'];

param.name_file_3rd_result = [param.name_file_3rd_result  '/'  ];
%%
param.name_simu = [ param_obs.assimilate  ...
    '/_DADuration_'  num2str(param_obs.SECONDS_OF_SIMU)  '_'];
switch param_obs.assimilate
    case 'real_data'
        switch param.type_data
            case 'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_truncated'
                param_obs.dt_PIV = 0.080833;
            case 'DNS100_inc3d_2D_2018_11_16_blocks_truncated'
                param_obs.dt_PIV = 0.05625;
        end
    case 'fake_real_data'
%         if strcmp(choice_n_subsample, 'auto_shanon')
            switch param.type_data
                case 'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_truncated'
                    param_obs.dt_PIV = 0.25;
                case 'DNS100_inc3d_2D_2018_11_16_blocks_truncated'
                    param_obs.dt_PIV = 0.05;
            end
%         else
%             error('unknown')
%         end
end
if param_obs.sub_sampling_PIV_data_temporaly
    param_obs.factor_of_PIV_time_subsampling = floor(param_obs.assimilation_period/param_obs.dt_PIV);
    param.name_simu = [param.name_simu  'ObsSubt_'  num2str(param_obs.factor_of_PIV_time_subsampling) '_'];
end
if param_obs.mask_obs
    param.name_simu = [param.name_simu  'ObsMaskyy_sub_'  num2str(param_obs.subsampling_PIV_grid_factor) ...
        '_from_'  num2str(param_obs.x0_index)  '_to_' ...
        num2str(param_obs.x0_index+param_obs.nbPoints_x*param_obs.subsampling_PIV_grid_factor) ...
        '_from_'  num2str(param_obs.y0_index)  '_to_' ...
        num2str(param_obs.y0_index+param_obs.nbPoints_y*param_obs.subsampling_PIV_grid_factor)  '_'];
else
    param_obs.x0_index =1;
    param_obs.y0_index =1;
    param_obs.subsampling_PIV_grid_factor = 1;
    param.name_simu = [param.name_simu  'no_mask_'];
end
if param.DA.init_centred_on_ref
    param.name_simu = [param.name_simu  'initOnRef_'];
end
param.name_simu = [param.name_simu  'beta_2_'  num2str(param.DA.beta_2)];
param.name_simu = [param.name_simu  '_nSimu_'  num2str(param_obs.n_simu)];
param.name_simu = [param.name_simu  '_nMut_'  num2str(param_obs.nb_mutation_steps)];
% param.name_simu = [param.name_simu  '_nSimu_'  num2str(param.n_simu)];
% param.name_simu = [param.name_simu  '_nMut_'  num2str(param.nb_mutation_steps)];

%%
param.name_file_3rd_result = [ param.name_file_3rd_result ...
    param.name_simu '/chronos.mat' ];
% param.name_file_3rd_result = [ param.folder_results ...
%     param.name_file_3rd_result 'chronos.mat' ];
        
        %%

% % if param.decor_by_subsampl.bool
% %     switch param.decor_by_subsampl.choice_n_subsample
% switch choice_n_subsample
%     case 'auto_shanon'
%         param.name_file_3rd_result=[ param.folder_results '3rdresult_' ...
%             param.type_data '_' num2str(param.nb_modes) '_modes_' ...
%             choice_n_subsample  ...
%             '_threshold_' num2str(param.decor_by_subsampl.spectrum_threshold) ...
%             param.decor_by_subsampl.test_fct ];
%     case {'lms', 'truncated', 'htgen', 'corr_time'} % corr_time for compatibility
%         param.name_file_3rd_result=[ param.folder_results '3rdresult_' ...
%             param.type_data '_' num2str(param.nb_modes) '_modes_' ...
%             choice_n_subsample '_'  ...
%             param.decor_by_subsampl.test_fct ];
% end
% % else
% %     param.name_file_3rd_result=[ param.folder_results '3rdresult_' ...
% %         param.type_data '_' num2str(param.nb_modes) '_modes_' ...
% %         ];
% % end
% param.name_file_3rd_result=[param.name_file_3rd_result '_fullsto'];
% mkdir(param.name_file_3rd_result)
% param.name_file_3rd_result=[param.name_file_3rd_result '\'];
% % if modal_dt
% %     param.name_file_3rd_result=[param.name_file_3rd_result '_modal_dt'];
% % end
% if modal_dt == 1
%     param.name_file_3rd_result=[param.name_file_3rd_result '_modal_dt'];
% elseif modal_dt == 2
%     param.name_file_3rd_result=[param.name_file_3rd_result '_real_dt'];
% end
% if ~ param.adv_corrected
%     param.name_file_3rd_result=[param.name_file_3rd_result '_no_correct_drift'];
% end
% % if param.decor_by_subsampl.no_subampl_in_forecast
% %     param.name_file_3rd_result=[param.name_file_3rd_result '_no_subampl_in_forecast'];
% % end
% if reconstruction
%     param.reconstruction=true;
%     param.name_file_3rd_result=[param.name_file_3rd_result '_reconstruction'];
% else
%     param.reconstruction=false;
% end
% param.name_file_3rd_result = [param.name_file_3rd_result '_integ_' stochastic_integration];
% if correlated_model
%     param.name_file_3rd_result = [param.name_file_3rd_result '_correlated'];
% end
% if estim_rmv_fv
%     param.name_file_3rd_result=[param.name_file_3rd_result '_estim_rmv_fv'];
% end
% if param.svd_pchol
%     param.name_file_3rd_result=[param.name_file_3rd_result '_svd_pchol'];    
% end
% if param.eq_proj_div_free == 2
%     param.name_file_3rd_result = [param.name_file_3rd_result '_DFSPN'];    
% end
% param.name_file_3rd_result=[param.name_file_3rd_result '.mat'];
% % save(param.name_file_3rd_result,'-v7.3');
% % % save(param.name_file_3rd_result);
% % clear C_deter C_sto L_deter L_sto I_deter I_sto
% % % if param.big_data
% %     toc;tic
% %     disp('3rd result saved');
% % % end


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

