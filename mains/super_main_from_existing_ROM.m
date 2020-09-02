function super_main_from_existing_ROM(...
    vect_nb_modes,type_data,v_threshold,vect_modal_dt,...
    no_subampl_in_forecast,vect_reconstruction,vect_adv_corrected,...
    decor_by_subsampl,eq_proj_div_free, svd_pchol,noise_type)
% Launch a set of simulations with a several set of parameters
% Especially several number of modes
%

% % Number of periods reconstructed
% nb_period_test = 9;% for DNS 300
% % nb_period_test = 5;%for DNS 3900

if nargin == 0
    init;
    global correlated_model
%     global choice_n_subsample;
    global stochastic_integration;
    global estim_rmv_fv;
    
    %% Number of modes for the the ROM
%     vect_nb_modes = [ 16 8 6 4 2]
    vect_nb_modes = [4]
    no_subampl_in_forecast = false;
    vect_adv_corrected = [ true false ]
    vect_svd_pchol = 2
%     vect_svd_pchol = svd_pchol
    
    %% Type of data
    % Other datasets (do not use)
    % % type_data = 'LES_3D_tot_sub_sample_blurred';
    % % type_data = 'incompact3d_wake_episode3_cut_truncated';
    % % type_data = 'inc3D_Re3900_blocks_truncated';
    % % type_data = 'inc3D_Re3900_blocks';
    % % type_data = 'incompact3D_noisy2D_40dt_subsampl';
    % % type_data = 'incompact3D_noisy2D_40dt_subsampl_truncated';
    % % type_data = 'inc3D_Re300_40dt_blocks_truncated';
    % % type_data = 'turb2D_blocks_truncated'
    % % type_data = 'test2D_blocks_truncated'
    % % type_data = 'DNS300_inc3d_3D_2017_04_02_blocks_truncated'
    % % type_data = 'inc3D_HRLESlong_Re3900_blocks_truncated'
    % % type_data = 'small_test_in_blocks_truncated'
    % % type_data = 'test_1_block'
    % % type_data = 'inc3D_Re3900_blocks';
    % % type_data = 'turb2D_blocks_truncated'
    
    % These 3D data ( Re 300) gives good results
    % type_data = 'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_truncated'
    
    % These 2D data ( Re 100) gives good results
    type_data = 'DNS100_inc3d_2D_2018_11_16_blocks_truncated'
    
    % Small dataset for debuging
%     type_data = 'incompact3D_noisy2D_40dt_subsampl_truncated';
    % % type_data = 'incompact3d_wake_episode3_cut_truncated';
    
    % Data of the Re 100 OpenFOAM simulation
    % type_data = 'DNS100_OpenFOAM_2D_2020_blocks_truncated'
    % type_data = 'DNS100_OpenFOAM_2D_2020_blocks'
    % type_data = 'DNS100_OpenFOAM_2D_2020'
    
    if strcmp(type_data((end-9):end),'_truncated')
        vect_reconstruction = false;
    else
        vect_reconstruction = true;
    end
    
    %% Important parameters
    v_threshold=nan
    vect_modal_dt=false
    
    % (specific) time sub-sampling (forced time-decorrelation of unresolved chronos)
    decor_by_subsampl.bool=true;
    % Choice of subsampling time step based on chronos
    decor_by_subsampl.test_fct='b';
    % Way the subsampling is done (in which part of the code)
    % (can be  'bt_decor' or  'a_estim_decor')
    decor_by_subsampl.meth='bt_decor';
    % Meth to choose the time sub-sampling
    % ('auto_shanon'=maxim frequency of resolved chronos)
    % ('corr_time' = autocorrelation time estimation of the unresolved chronos)
    decor_by_subsampl.choice_n_subsample='htgen2'; % 'htgen' 'auto_shanon' 'lms'
    % decor_by_subsampl.choice_n_subsample = 'corr_time';
    decor_by_subsampl.bug_sampling = false ;
    
    % Stochastic integration path : 'Ito' or 'Str'
    stochastic_integration = 'Ito';
    
    % Definition of global variable to manage methods more easily
    choice_n_subsample = decor_by_subsampl.choice_n_subsample;
    
    % During the noise covariance estimation,
    % remove the finite-variation part of the chronos
    estim_rmv_fv = true;
    
    % Projection on the free-divergence-function space
    % 0 : no projection / 1 : projection of deterministic terms
    %  / 2 :projection of noise terms
    eq_proj_div_free = 2;
    
    correlated_model = false
else    
%     global choice_n_subsample;
    global stochastic_integration;
    global estim_rmv_fv;
    global correlated_model
    vect_svd_pchol = svd_pchol;
end

% % v_threshold=[1 10]/1000;
% % v_threshold=1e-3;
% if strcmp(type_data,'incompact3d_wake_episode3_cut_truncated')
%     % Number of periods reconstructed
%     nb_period_test = 9;% for DNS 300
% elseif strcmp(type_data,'LES_3D_tot_sub_sample_blurred')
%     % Number of periods reconstructed
%     nb_period_test = 5;%for DNS 3900
% else
    nb_period_test = nan;
% end

nb_modes_max = max(vect_nb_modes);

for svd_pchol=vect_svd_pchol
for modal_dt=vect_modal_dt
    for q=1:length(v_threshold)
        % parfor q=1:length(v_threshold)
        threshold=v_threshold(q);
        decor_by_subsampl_temp=decor_by_subsampl;
        decor_by_subsampl_temp.spectrum_threshold=threshold;
        for adv_corrected = vect_adv_corrected
            for reconstruction = vect_reconstruction
                % for k=vect_nb_modes
                for kk=1:length(vect_nb_modes)
                    k=vect_nb_modes(kk);
                    %     for k=nb_modes_min:2:nb_modes_max
                    main_from_existing_ROM(k,threshold,type_data,...
                        nb_period_test,...
                        no_subampl_in_forecast,reconstruction,...
                        adv_corrected,modal_dt,decor_by_subsampl_temp,...
                        svd_pchol,eq_proj_div_free,noise_type)
%                         adv_corrected,modal_dt,decor_by_subsampl.test_fct,...
%                         svd_pchol,eq_proj_div_free)
                    %         main_full_sto_vect_modal_dt_2nd_res(k,v_threshold(q))
                end
                %% Save plot
                %         folder_results = [ pwd '/resultats/current_results/'];
                %         current_pwd = pwd; cd ..
                %         folder_data = [ pwd '/data/' ];
                %         cd(current_pwd);
                %         % folder_results = ['/Users/Resseguier/Documents/MATLAB/POD/all/resultats/current_results/'];
                %         % folder_data = '/Users/Resseguier/Documents/MATLAB/POD/data/';
                %         % folder_results = ['/Users/vressegu/Documents/matlab/POD-NS_Stochastique/current_used/' ...
                %         %         'all/resultats/current_results/'];
                %         % %     param.folder_results =  [ pwd '/resultats/current_results/'];
                %
                %         threshold = num2str(q);
                %         iii = (threshold =='.');
                %         threshold(iii)='_';
                %
                %         str = ['print -depsc ' folder_results type_data '_sum_modes_n=' ...
                %             num2str(nb_modes_max) '_threshold_' threshold ...
                %             '_fullsto'];
                %         if vect_modal_dt
                %             str =[ str '_vect_modal_dt'];
                %         end
                %         if ~ adv_corrected
                %             str =[ str '_vect_modal_dt_NoAdvCorect'];
                %         end
                %         str =[ str '.eps'];
                %         eval(str);
                %         %     eval( ['print -depsc ' folder_results type_data '_sum_modes_n=' ...
                %         %         num2str(nb_modes_max) '_threshold_' num2str(v_threshold(q)) ...
                %         %         '_fullsto_vect_modal_dt.eps']);
                %         % %     eval( ['print -depsc ' folder_results 'sum_modes_n=' ...
                %         % %         num2str(nb_modes_max) '_threshold_' num2str(v_threshold(q)) '_fullsto_vect_modal_dt_test_basis.eps']);
                
            end
        end
    end
end
end

%% Plots
super_main_from_existing_ROM_Simulation(...
    vect_nb_modes,type_data,v_threshold,vect_modal_dt,...
    no_subampl_in_forecast,vect_reconstruction,vect_adv_corrected,...
    decor_by_subsampl,vect_svd_pchol,eq_proj_div_free,noise_type)
%     decor_by_subsampl.test_fct,vect_svd_pchol,eq_proj_div_free)

