% Plot the energy and variance over time
clear all;
close all;

%% Parameters of the program
% param.type_data = 'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_truncated';
% param.type_data = 'DNS300_inc3d_3D_2017_04_02_blocks';
% param.type_data = 'turb2D_blocks';
param.type_data = 'DNS100_inc3d_2D_2018_11_16_blocks_truncated'
% param.type_data = 'incompact3d_wake_episode3_cut_truncated';

switch param.type_data
    case {'DNS100_inc3d_2D_2018_11_16_blocks_truncated'}
        param.decor_by_subsampl.spectrum_threshold = 1e-6
        vect_nb_modes = [ 2 4 6 8 16 ]
        %         param.nb_modes = 16
        param.adv_corrected = true
        %         param.adv_corrected = false
        modal_dt = 1
        time_plot = 40
%         if param.nb_modes <= 8
%             time_plot = 40
%         else
%             time_plot = 10
%         end
    case 'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_truncated'
        param.decor_by_subsampl.spectrum_threshold = 1e-4
%         param.nb_modes = 2
        vect_nb_modes = [ 2 4 6 8 16 ]
        param.adv_corrected = false
        modal_dt = 1
%         modal_dt = 0
        time_plot = 20
    case {'incompact3d_wake_episode3_cut',...
            'incompact3d_wake_episode3_cut_truncated'}
        param.decor_by_subsampl.spectrum_threshold = 1e-6
%         param.nb_modes = 16
        vect_nb_modes = [ 2 4 8 16 ]
        param.adv_corrected = true
        modal_dt = false
        time_plot = nan
    otherwise
        error('unknown case')
end
reconstruction = false;
param.a_time_dependant = false;
param.decor_by_subsampl.bool=true;
param.decor_by_subsampl.test_fct='b';
param.decor_by_subsampl.meth='bt_decor';
param.decor_by_subsampl.choice_n_subsample='auto_shanon';
param.decor_by_subsampl.no_subampl_in_forecast = false;

% folder_results = 'F:\GitHub\PODFS/resultats/current_results/';
param.folder_results = [ pwd '/resultats/current_results/'];
current_pwd = pwd; cd ..
param.folder_data = [ pwd '/data/' ];
cd(current_pwd);

param_REFF = param;
for k=vect_nb_modes
    param = param_REFF;
    param.nb_modes = k
    if strcmp(param.type_data,'DNS100_inc3d_2D_2018_11_16_blocks_truncated') && k > 8
        time_plot = 10
    end
    %% Loading the correct results file and important parameters included in it
    
    param = fct_name_2nd_result(param,modal_dt,reconstruction);
    param_ref = param;
    load(param.name_file_2nd_result,'bt_MCMC', 'param', ...
        'I_deter', 'L_deter', 'C_deter', ...
        'I_sto', 'L_sto', 'C_sto', 'pchol_cov_noises', 'F1', 'F2');
    param.name_file_2nd_result = param_ref.name_file_2nd_result;
    param.adv_corrected = param_ref.adv_corrected;
    
    % results_file = [folder_results '2ndresult_' type_data '_truncated_' num2str(nb_modes) ...
    %     '_modes__a_cst__decor_by_subsampl_bt_decor_choice_auto_shanon_threshold_' ...
    %     threshold 'fct_test_b_fullsto_modal_dt.mat'];
    % load(results_file)
    % % dt = 3600*24;
    % clearvars -except bt_MCMC type_data folder_results nb_modes current_pwd
    % [N_t, nb_modes, N_particules] = size(bt_MCMC);
    
    if ~isnan(time_plot)
        N_plot = floor( time_plot/param.dt);
        bt_MCMC = bt_MCMC(1:N_plot,:,:);
    end
    
    Cov_noises = pchol_cov_noises * pchol_cov_noises';
    if ~exist('F1','var') || ~exist('F2','var')
        F1 = nan; F2 = nan;
    end
    
    %% Plots
    plot_modal_nrj(bt_MCMC, param, ...
        I_deter, L_deter, C_deter, ...
        I_sto, L_sto, C_sto, Cov_noises, F1, F2);
    
    %% Save plot
    
    
    %% Save plot
    threshold = num2str(param.decor_by_subsampl.spectrum_threshold);
    iii = (threshold =='.');
    threshold(iii)='_';
%     str = ['print -depsc ' param.folder_results param.type_data ...
    str = ['print -dpng ' param.folder_results param.type_data ...
        '_energy_fluxes_n=' ...
        num2str(param.nb_modes) '_threshold_' threshold ...
        '_fullsto'];
    if modal_dt == 1
        str =[ str '_modal_dt'];
    elseif modal_dt == 2
        str =[ str '_real_dt'];
    end
    if ~ param.adv_corrected
        str =[ str '_NoAdvCorect'];
    end
    if reconstruction
        str =[ str '_reconstruction'];
    else
        str =[ str '_forecast'];
    end
%     %                 str =[ str '.png'];
%     str1 =[ str '.eps'];
%     str2 =[ str '_filtered.eps'];
    str2 =[ str '_filtered'];
    
    str1 =[ str '.png'];
    str2 =[ str2 '.png'];
%     str1 =[ str '.eps'];
%     str2 =[ str2 '.eps'];
    
    figure(1)
    str1
    drawnow
    pause(1)
    eval(str1);
    figure(2)
    str2
    drawnow
    pause(1)
    eval(str2);
end
