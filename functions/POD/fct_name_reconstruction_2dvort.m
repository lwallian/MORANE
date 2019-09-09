function param = fct_name_reconstruction_2dvort(...
    param,modal_dt,reconstruction,name_simu)
% Create the name of the file where the reconstruction of U are saved
%

global stochastic_integration;
global estim_rmv_fv;
global choice_n_subsample;
global correlated_model;

switch param.data_assimilation
    case {0,1}
        folder_data = param.folder_data;
    case 2
        folder_data = param.folder_data_PIV;
end
param.name_file_Reconstruction_omega =[ folder_data 'Reconstruction_2dvort_' ...
    param.type_data '_subSampl_' num2str(param.decor_by_subsampl.n_subsampl_decor) '/'];
if ~strcmp(name_simu,'ref')
    % if strcmp(name_simu,'ref')
    %     param.name_file_Reconstruction_omega=[ folder_data 'Reconstruction_2dvort_' ...
    %         param.type_data '_' num2str(param.nb_modes) '_modes_'  ...
    %         'REF' ...
    
    param.name_file_Reconstruction_omega=[ param.name_file_Reconstruction_omega ...
        '_' num2str(param.nb_modes) '_modes_'  ...
        choice_n_subsample ];
%     param.name_file_Reconstruction_omega=[ folder_data ...
%         param.type_data '_' num2str(param.nb_modes) '_modes_'  ...
%         choice_n_subsample ];
    if strcmp(choice_n_subsample, 'auto_shanon')
        param.name_file_Reconstruction_omega = [param.name_file_Reconstruction_omega  '_threshold_' ...
            num2str(param.decor_by_subsampl.spectrum_threshold)];
    end
    param.name_file_Reconstruction_omega = [param.name_file_Reconstruction_omega  param.decor_by_subsampl.test_fct  '/'];
    if strcmp(choice_n_subsample, 'auto_shanon') && modal_dt
        param.name_file_Reconstruction_omega = [param.name_file_Reconstruction_omega  '_modal_dt'];
    end
    if ~ param.adv_corrected
        param.name_file_Reconstruction_omega = [param.name_file_Reconstruction_omega  '_no_correct_drift'];
    end
    
    
    param.name_file_Reconstruction_omega = [param.name_file_Reconstruction_omega '_integ_' stochastic_integration];
    if correlated_model
        param.name_file_Reconstruction_omega = [param.name_file_Reconstruction_omega '_correlated'];
    end
    if estim_rmv_fv
        param.name_file_Reconstruction_omega=[param.name_file_Reconstruction_omega '_estim_rmv_fv'];
    end
    if param.svd_pchol
        param.name_file_Reconstruction_omega=[param.name_file_Reconstruction_omega '_svd_pchol'];
    end
    if param.eq_proj_div_free == 2
        param.name_file_Reconstruction_omega = [param.name_file_Reconstruction_omega '_DFSPN'];
    end
    % %         param.name_file_Reconstruction_omega = [param.name_file_Reconstruction_omega  '_integ_Ito'];
    % % if param.svd_pchol
    % %     param.name_file_Reconstruction_omega = [param.name_file_Reconstruction_omega  '_svd_pchol'];
    % % end
    % % param.name_file_Reconstruction_omega = [param.name_file_Reconstruction_omega  '/'  param_obs.assimilate  ...
    % %     '/_DADuration_'  num2str(param_obs.SECONDS_OF_SIMU)  '_'];
    %
    % param.name_file_Reconstruction_omega = [param.name_file_Reconstruction_omega  '/'  ];
    %%
    % param.name_simu = [ param_obs.assimilate  ...
    %     '/_DADuration_'  num2str(param_obs.SECONDS_OF_SIMU)  '_'];
    % switch param_obs.assimilate
    %     case 'real_data'
    %         switch param.type_data
    %             case 'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_truncated'
    %                 param_obs.dt_PIV = 0.080833;
    %             case 'DNS100_inc3d_2D_2018_11_16_blocks_truncated'
    %                 param_obs.dt_PIV = 0.05625;
    %         end
    %     case 'fake_real_data'
    %         if strcmp(choice_n_subsample, 'auto_shanon')
    %             switch param.type_data
    %                 case 'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_truncated'
    %                     param_obs.dt_PIV = 0.25;
    %                 case 'DNS100_inc3d_2D_2018_11_16_blocks_truncated'
    %                     param_obs.dt_PIV = 0.05;
    %             end
    %         else
    %             error('unknown')
    %         end
    % end
    % if param_obs.sub_sampling_PIV_data_temporaly
    %     param_obs.factor_of_PIV_time_subsampling = floor(param_obs.assimilation_period/param_obs.dt_PIV);
    %     param.name_simu = [param.name_simu  'ObsSubt_'  num2str(param_obs.factor_of_PIV_time_subsampling) '_'];
    % end
    % if param_obs.mask_obs
    %     param.name_simu = [param.name_simu  'ObsMaskyy_sub_'  num2str(param_obs.subsampling_PIV_grid_factor) ...
    %         '_from_'  num2str(param_obs.x0_index)  '_to_' ...
    %         num2str(param_obs.x0_index+param_obs.nbPoints_x*param_obs.subsampling_PIV_grid_factor) ...
    %         '_from_'  num2str(param_obs.y0_index)  '_to_' ...
    %         num2str(param_obs.y0_index+param_obs.nbPoints_y*param_obs.subsampling_PIV_grid_factor)  '_'];
    % else
    %     param_obs.x0_index =1;
    %     param_obs.y0_index =1;
    %     param_obs.subsampling_PIV_grid_factor = 1;
    %     param.name_simu = [param.name_simu  'no_mask_'];
    % end
    % if param.DA.init_centred_on_ref
    %     param.name_simu = [param.name_simu  'initOnRef_'];
    % end
    % param.name_simu = [param.name_simu  'beta_2_'  num2str(param.DA.beta_2)];
    %%
    % param.name_file_Reconstruction_omega = [ param.name_file_Reconstruction_omega ...
    %     param.name_simu '/chronos.mat' ];
    
    %%
    %%
    
    % if strcmp(name_simu,'ref')
    %     %             param.name_file_Reconstruction_omega=[ ...
    %     %             'C:\Users\valentin.resseguier\Google_Drive\boulot\scalian\Red_LUM\plots\' ...
    %     %             'Reconstruction_2dvort_' ...
    %     param.name_file_Reconstruction_omega=[ folder_data 'Reconstruction_2dvort_' ...
    %         param.type_data '_' num2str(param.nb_modes) '_modes_'  ...
    %         'REF' ...
    %         '_decor_by_subsampl_' num2str(param.decor_by_subsampl.meth) ...
    %         '_choice_' num2str(param.decor_by_subsampl.choice_n_subsample)  ...
    %         '_threshold_' num2str(param.decor_by_subsampl.spectrum_threshold) ...
    %         'fct_test_' param.decor_by_subsampl.test_fct ];
    %     if param.decor_by_subsampl.no_subampl_in_forecast
    %         param.name_file_Reconstruction_omega=[param.name_file_Reconstruction_omega '_no_subampl_in_forecast'];
    %     end
    %     if reconstruction
    %         param.reconstruction=true;
    %         param.name_file_Reconstruction_omega=[param.name_file_Reconstruction_omega '_reconstruction'];
    %     else
    %         param.reconstruction=false;
    %     end
    %     param.name_file_Reconstruction_omega = [ param.name_file_Reconstruction_omega '/' ];
    % else
    %     if param.a_time_dependant
    %         dependance_on_time_of_a = '_a_time_dependant_';
    %     else
    %         dependance_on_time_of_a = '_a_cst_';
    %     end
    %     if param.decor_by_subsampl.bool
    %         if strcmp(dependance_on_time_of_a,'a_t')
    %             char_filter = [ '_on_' param.type_filter_a ];
    %         else
    %             char_filter = [];
    %         end
    %         %                 param.name_file_Reconstruction_omega=[ ...
    %         %             'C:\Users\valentin.resseguier\Google_Drive\boulot\scalian\Red_LUM\plots\' ...
    %         %             'Reconstruction_2dvort_' ...
    %         param.name_file_Reconstruction_omega=[ folder_data 'Reconstruction_2dvort_' ...
    %             param.type_data '_' num2str(param.nb_modes) '_modes_' ...
    %             dependance_on_time_of_a char_filter ...
    %             '_decor_by_subsampl_' num2str(param.decor_by_subsampl.meth) ...
    %             '_choice_' num2str(param.decor_by_subsampl.choice_n_subsample)  ...
    %             '_threshold_' num2str(param.decor_by_subsampl.spectrum_threshold) ...
    %             'fct_test_' param.decor_by_subsampl.test_fct ];
    %     else
    %         %                 param.name_file_Reconstruction_omega=[ ...
    %         %             'C:\Users\valentin.resseguier\Google_Drive\boulot\scalian\Red_LUM\plots\' ...
    %         %             'Reconstruction_2dvort_' ...
    %         param.name_file_Reconstruction_omega=[ param.folder_data 'Reconstruction_2dvort_' ...
    %             param.type_data '_' num2str(param.nb_modes) '_modes_' ...
    %             dependance_on_time_of_a ];
    %     end
    %     param.name_file_Reconstruction_omega=[param.name_file_Reconstruction_omega '_fullsto'];
    %     % if modal_dt
    %     %     param.name_file_Reconstruction_omega=[param.name_file_Reconstruction_omega '_modal_dt'];
    %     % end
    %     if modal_dt == 1
    %         param.name_file_Reconstruction_omega=[param.name_file_Reconstruction_omega '_modal_dt'];
    %     elseif modal_dt == 2
    %         param.name_file_Reconstruction_omega=[param.name_file_Reconstruction_omega '_real_dt'];
    %     end
    %     if ~ param.adv_corrected
    %         param.name_file_Reconstruction_omega=[param.name_file_Reconstruction_omega '_no_correct_drift'];
    %     end
    %     if param.decor_by_subsampl.no_subampl_in_forecast
    %         param.name_file_Reconstruction_omega=[param.name_file_Reconstruction_omega '_no_subampl_in_forecast'];
    %     end
    %     if reconstruction
    %         param.reconstruction=true;
    %         param.name_file_Reconstruction_omega=[param.name_file_Reconstruction_omega '_reconstruction'];
    %     else
    %         param.reconstruction=false;
    %     end
    % %     param.name_file_Reconstruction_omega = [ param.name_file_Reconstruction_omega '/' ...
    % %         name_simu '/'];
    % end
end
if isfield(param,'param_obs')
    if param.param_obs.no_noise
        name_simu = [name_simu '_noNoise'];
    end
    param.name_file_Reconstruction_omega = [ param.name_file_Reconstruction_omega '/' ...
        name_simu];
end
param.name_file_Reconstruction_omega = [ param.name_file_Reconstruction_omega '/'];
mkdir(param.name_file_Reconstruction_omega);

