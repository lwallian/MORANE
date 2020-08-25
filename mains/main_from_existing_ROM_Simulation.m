function main_from_existing_ROM_Simulation(type_data,nb_modes,...
    threshold,no_subampl_in_forecast,reconstruction,adv_corrected,...
    modal_dt,decor_by_subsampl,svd_pchol,eq_proj_div_free,plot_EV_noise)
%     modal_dt,test_fct,svd_pchol,eq_proj_div_free,plot_EV_noise)
% Load simulation results, estimate modal time step by Shanon
% and compare it with modal Eddy Viscosity ROM and
% tuned version of the loaded results
%

clear param bt_forecast_sto bt_forecast_deter bt_tot
tic

%% Parameters choice

% Plots to do
plot_deterministic=true; % deterministic POD-Galerkin
plot_EV=true; % estimated Eddy Visocvity
% plot_EV_noise=true; % estimated Eddy Visocvity
plot_tuned=false; % estimated corrective coefficients

if nargin < 7
    modal_dt=false; % different time step (infered by Shanon criterion) for the different modes
    switch type_data
        case 'incompact3d_wake_episode3_cut_truncated'
            modal_dt=false;
        case 'inc3D_Re3900_blocks_truncated'
            modal_dt=true;
    end
end


plot_each_mode=false;

% Threshold of the Chonos spectrum, used to choice the time step
% threshold=0.001; % 0.001 or 0.01 for LES 3900 (or inc3D 3900)
% threshold=0.005; % for LES 3900
% threshold=0.0005; % for inc3D episode 3
% threshold=0.00014; % for inc3D 3900
% threshold=0.000135; % for inc3D 3900

% Number of particle for the MCMC simulation
% param_ref2.N_particules=100;
% % param_ref2.N_particules=1000;
% % param_ref.N_particules=2;
% % param.N_particules=2;
% % % param.N_particules=1000;

% Rate of increase of the time step to simulate accurately the SDE
% param_ref2.n_simu = 100;
% % param_ref.n_simu = 1;
% % n_simu = 1;
% % % n_simu = 100;

% On which function the Shanon ctriterion is used
% test_fct='b'; % 'b' is better than db

% Learning duration
% period_estim=3;
% % p_estim=13;
% % N_estim=842;
% coef_correctif_estim.learning_time='N_estim'; % 'quarter' or 'all'
% % coef_correctif_estim.learning_time='quarter'; % 'quarter' or 'all'

% % Type of data
%
% % These 3D data give good results
% % They are saved in only one file
% % (~ 250 time step)
% % type_data = 'LES_3D_tot_sub_sample_blurred';
% % type_data = 'incompact3d_wake_episode3_cut';
% type_data = 'inc3D_Re3900_blocks';
% %     type_data = 'incompact3D_noisy2D_40dt_subsampl';

% These 3D data are bigger, since the spatial grid is thinner
% and the number of time step is bigger
% (~ 2000 time step)
% They are saved in different files
% The internship should try to use these data
%     type_data = 'inc3D_Re3900_blocks';

% Number of POD modes
if nargin == 0
    nb_modes = 2;
end

% On which function the Shanon ctriterion is used
if nargin < 8 
    test_fct = 'b';
end
if nargin < 9 
    svd_pchol = false;
end
% if nargin < 9 
%     test_fct = 'b';
% end
% if nargin < 10 
%     svd_pchol = false;
% end
param_ref2.decor_by_subsampl = decor_by_subsampl;
% param_ref2.decor_by_subsampl.test_fct = test_fct;
param_ref2.svd_pchol=svd_pchol;

% % On which function the Shanon ctriterion is used
% decor_by_subsampl.test_fct = 'b';

%% Parameters already chosen
% Do not modify the following lines

coef_correctif_estim.learn_coef_a=true; % true false
coef_correctif_estim.type_estim='vector_b'; % 'scalar' 'vector_z' 'vector_b' 'matrix'
coef_correctif_estim.beta_min=-inf; % -inf 0 1
coef_correctif_estim.nb_modes_used=eval('nb_modes'); % 2 eval('nb_modes') for learning the coefficient

folder_results = [ pwd '/resultats/current_results/'];
current_pwd = pwd; cd ..
folder_data = [ pwd '/data/' ];
cd(current_pwd);
% folder_results = ['/Users/Resseguier/Documents/MATLAB/POD/all/resultats/current_results/'];
% folder_data = '/Users/Resseguier/Documents/MATLAB/POD/data/';
% folder_results = ['/Users/vressegu/Documents/matlab/POD-NS_Stochastique/current_used/' ...
%     'all/resultats/current_results/'];
%     param.folder_results =  [ pwd '/resultats/current_results/'];
param_ref2.folder_results=folder_results;
param_ref2.folder_data =folder_data ;

if nargin > 0
    plot_each_mode=false;
end

modal_dt_ref = modal_dt;
folder_results_ref = folder_results;
folder_data_ref = folder_data;

%% Get data

a_t='_a_cst_';

% global choice_n_subsample;
global stochastic_integration;
global estim_rmv_fv;
global correlated_model;

param_ref2.decor_by_subsampl = decor_by_subsampl;
% param_ref2.decor_by_subsampl.spectrum_threshold = threshold;
param_ref2.type_data = type_data;
param_ref2.nb_modes = nb_modes;
param_ref2.adv_corrected = adv_corrected;
% param_ref2.decor_by_subsampl.choice_n_subsample = choice_n_subsample;
param_ref2.eq_proj_div_free = eq_proj_div_free;

if (~ strcmp(param_ref2.decor_by_subsampl.choice_n_subsample,'auto_shanon'))
    modal_dt = true;
end
param_ref2 = fct_name_2nd_result_new(param_ref2,modal_dt,reconstruction);
file_res_2nd_res = param_ref2.name_file_2nd_result;
file_res_2nd_res
erase(file_res_2nd_res,'_modal_dt')
if ~ (exist(file_res_2nd_res,'file') == 2)
    if (~ strcmp(choice_n_subsample,'auto_shanon')) && ...
            (exist(erase(file_res_2nd_res,'_modal_dt'),'file') == 2)
        file_res_2nd_res = erase(file_res_2nd_res,'_modal_dt');
    else        
        switch choice_n_subsample
            case 'auto_shanon'
                file_res_2nd_res=[ folder_results '2ndresult_' type_data '_' num2str(nb_modes) '_modes_' ...
                    a_t '_decor_by_subsampl_bt_decor_choice_auto_shanon_threshold_' ...
                    num2str(threshold) ...
                    'fct_test_' test_fct ];
            case 'lms'
                file_res_2nd_res=[ folder_results '2ndresult_' type_data '_' num2str(nb_modes) '_modes_' ...
                    a_t '_decor_by_subsampl_bt_decor_choice_lms_' ...
                    'fct_test_' test_fct];
            case 'truncated'
                file_res_2nd_res=[ folder_results '2ndresult_' type_data '_' num2str(nb_modes) '_modes_' ...
                    a_t '_decor_by_subsampl_bt_decor_choice_truncated_' ...
                    'fct_test_' test_fct];
            case 'htgen'
                file_res_2nd_res=[ folder_results '2ndresult_' type_data '_' num2str(nb_modes) '_modes_' ...
                    a_t '_decor_by_subsampl_bt_decor_choice_htgen_' ...
                    'fct_test_' test_fct];
            otherwise
                error('unknown case');
        end
        
        file_res_2nd_res=[file_res_2nd_res '_fullsto'];
        if modal_dt == 1
            file_res_2nd_res=[file_res_2nd_res '_modal_dt'];
        elseif modal_dt == 2
            file_res_2nd_res=[file_res_2nd_res '_real_dt'];
        end
        if ~ adv_corrected
            file_res_2nd_res=[file_res_2nd_res '_no_correct_drift'];
        end
        if no_subampl_in_forecast
            file_res_2nd_res=[file_res_2nd_res '_no_subampl_in_forecast'];
        end
        if reconstruction
            file_res_2nd_res=[file_res_2nd_res '_reconstruction'];
        end
        if correlated_model
            file_res_2nd_res = [file_res_2nd_res '_correlated_'];
        end
        file_res_2nd_res_save = file_res_2nd_res;
        file_res_2nd_res=[file_res_2nd_res '_integ_' stochastic_integration];
        if estim_rmv_fv
            file_res_2nd_res=[file_res_2nd_res '_estim_rmv_fv'];
            param.estim_rmv_fv = true;
        end
        if svd_pchol
            file_res_2nd_res=[file_res_2nd_res '_svd_pchol'];
        end
        
        % Annoying cases
        file_res_2nd_res=[file_res_2nd_res '.mat'];
        if (~(exist(file_res_2nd_res,'file') == 2)) ...
                && strcmp(stochastic_integration,'Ito')
            file_res_2nd_res = file_res_2nd_res_save;
            if estim_rmv_fv
                file_res_2nd_res=[file_res_2nd_res '_estim_rmv_fv'];
                param.estim_rmv_fv = true;
            end
            if svd_pchol
                file_res_2nd_res=[file_res_2nd_res '_svd_pchol'];
            end
            file_res_2nd_res=[file_res_2nd_res '.mat'];
        else
            clear file_res_2nd_res_save;
        end
        if (~(exist(file_res_2nd_res,'file') == 2)) && ...
                strcmp(choice_n_subsample,'auto_shanon')
            file_res_2nd_res = erase(file_res_2nd_res,'_modal_dt');
            if (~(exist(file_res_2nd_res,'file') == 2)) ...
                    && strcmp(stochastic_integration,'Ito')
                file_res_2nd_res = erase(file_res_2nd_res,'_integ_Ito');
            end            
        end
            
    end
end
load(file_res_2nd_res)
if (~ strcmp(param_ref2.decor_by_subsampl.choice_n_subsample,'auto_shanon'))
    modal_dt = false;
end

if reconstruction
    param.reconstruction=true;
else
    param.reconstruction=false;
end

% % file_res=[ folder_results '2ndresult_' type_data '_' num2str(nb_modes) '_modes_' ...
% file_res=[ folder_results '1stresult_' type_data '_' num2str(nb_modes) '_modes_' ...
%     a_t '_decor_by_subsampl_bt_decor_choice_auto_shanon_threshold_' ...
%     num2str(threshold) ...
%     'fct_test_' test_fct ];
% if exist('period_estim','var')
%     file_res=[file_res '_p_estim_' num2str(period_estim)];
% end
% file_res=[file_res '_fullsto'];
% file_res=[ file_res '.mat'];
% save(file_res)
% % load([ folder_results '2ndresult_' type_data '_' num2str(nb_modes) '_modes_' ...
% %     a_t '_decor_by_subsampl_bt_decor_choice_auto_shanon_threshold_' ...
% %     num2str(threshold) ...
% %     'fct_test_' test_fct '.mat']);
%%

folder_results = folder_results_ref;
param.folder_results = folder_results;
folder_data = folder_data_ref;
param.folder_data = folder_data;
modal_dt = modal_dt_ref;
modal_dt

param.decor_by_subsampl=decor_by_subsampl;

folder_data = param_ref2.folder_data;
folder_results = param_ref2.folder_results;
% folder_data = '/Users/vressegu/Documents/matlab/POD-NS_Stochastique/current_used/data/';
% folder_results = ['/Users/vressegu/Documents/matlab/POD-NS_Stochastique/current_used/' ...
%     'all/resultats/current_results/'];

%     param.folder_results =  [ pwd '/resultats/current_results/'];
big_data=false;
plot_bts=true;

% coef_correctif_estim=coef_correctif_estim_ref;

param.folder_data = folder_data;
param.folder_results = folder_results;
param.big_data=big_data;
param.plot_bts=plot_bts;
param.coef_correctif_estim=coef_correctif_estim;

% if strcmp(param.type_data, 'inc3D_Re3900_blocks')
%     param.N_test = ceil(10*5/param.dt);
%     warning('simulation on only 10 periods')
% end

param.folder_results=param_ref2.folder_results;

% param.N_particules=param_ref2.N_particules;
% n_simu=param_ref2.n_simu;


if ~isfield(struct_bt_MCMC.tot,'one_realiz')
    struct_bt_MCMC.tot.one_realiz = bt_MCMC(:,:,1);
end

if ~isfield(struct_bt_MCMC,'qtl')
    % BETA : confidence interval
    struct_bt_MCMC.qtl = fx_quantile(bt_MCMC, 0.025, 3);
    struct_bt_MCMC.diff = fx_quantile(bt_MCMC, 0.975, 3) - struct_bt_MCMC.qtl;
    save(file_res_2nd_res,'struct_bt_MCMC','-append');
end

%% Eddy viscosity solutions

param.plot_EV_noise = plot_EV_noise;
if plot_EV_noise
    param.plot.plot_EV= plot_EV;
    file_EV=[ param.folder_results 'EV_result_' param.type_data ...
        '_' num2str(param.nb_modes) '_modes'];
    file_EV=[file_EV '_noise.mat'];
    load(file_EV,'param_deter',...
        'bt_forecast_MEV','bt_forecast_EV','bt_forecast_NLMEV');
%     %     load(file_EV,'param_deter',...
%     %         'bt_forecast_deter',...
%     %         'bt_forecast_MEV','bt_forecast_EV','bt_forecast_NLMEV');
%     %     if modal_dt ~= 1
%     bt_forecast_MEV = ...
%         bt_forecast_MEV(1:param.decor_by_subsampl.n_subsampl_decor:end,:,:);
%     bt_forecast_EV = ...
%         bt_forecast_EV(1:param.decor_by_subsampl.n_subsampl_decor:end,:,:);
    
%     bt_forecast_EV= reshape( bt_forecast_EV, ...
%         [param.N_test+1 param.nb_modes param.N_particules]);
%     bt_forecast_MEV= reshape( bt_forecast_MEV, ...
%         [param.N_test+1 param.nb_modes param.N_particules]);
    bt_forecast_EV_noise = bt_forecast_EV; clear bt_forecast_EV
    bt_forecast_MEV_noise = bt_forecast_MEV; clear bt_forecast_MEV
    bt_forecast_MEV_noise = bt_forecast_EV_noise;
    %     end
    clear bt_forecast_EV_noise
    
    struct_bt_MEV_noise.tot.mean = mean(bt_forecast_MEV_noise, 3);
    struct_bt_MEV_noise.tot.var = var(bt_forecast_MEV_noise, 0, 3);
    struct_bt_MEV_noise.tot.one_realiz = bt_forecast_MEV_noise(:, :, 1);
    % BETA : confidence interval
    struct_bt_MEV_noise.qtl = fx_quantile(bt_forecast_MEV_noise, 0.025, 3);
    struct_bt_MEV_noise.diff = fx_quantile(bt_forecast_MEV_noise, 0.975, 3) ...
        - struct_bt_MEV_noise.qtl;
else
    struct_bt_MEV_noise = nan;
    bt_forecast_MEV_noise = nan;
end

if plot_EV
    param.plot.plot_EV= plot_EV;
    file_EV=[ param.folder_results 'EV_result_' param.type_data ...
        '_' num2str(param.nb_modes) '_modes'];
    file_EV=[file_EV '.mat'];
    load(file_EV,'param_deter',...
        'bt_forecast_MEV','bt_forecast_EV','bt_forecast_NLMEV');
    %     load(file_EV,'param_deter',...
    %         'bt_forecast_deter',...
    %         'bt_forecast_MEV','bt_forecast_EV','bt_forecast_NLMEV');
    %     if modal_dt ~= 1
    bt_forecast_MEV = bt_forecast_EV;
    %     end
    clear bt_forecast_EV
%     bt_forecast_MEV = ...
%         bt_forecast_MEV(1:param.decor_by_subsampl.n_subsampl_decor:end,:);
else
    bt_forecast_MEV = nan;
end


%% Plots of the reconstructed Chronos
if plot_bts
    param.folder_data =param_ref2.folder_data ;
    
    param.plot.plot_deter=plot_deterministic;
    param.plot.plot_EV=plot_EV;
    param.plot.plot_tuned=plot_tuned;
    param.plot_modal_dt = false;
    %     param.plot_modal_dt = plot_modal_dt;
    
    zzz = zeros(size(bt_tot));
    %     param.plot.plot_EV = false;
    if ~ param.plot.plot_EV
        bt_forecast_MEV = zzz;
    end
    
    %     if param_ref.plot_each_mode
    plot_bt_MCMC(param,bt_tot,bt_tot,...
        bt_tot, bt_tot, bt_forecast_deter,...
        bt_forecast_MEV,struct_bt_MEV_noise,bt_forecast_sto,bt_tot,struct_bt_MCMC);
%         bt_forecast_MEV,bt_forecast_sto,bt_forecast_sto,bt_tot,struct_bt_MCMC)

    %     plot_bt_MCMC(param,bt_tot,bt_tot,...
    %         bt_tot, bt_tot, bt_forecast_deter,...
    %         bt_tot,bt_forecast_sto,bt_forecast_sto,bt_tot,struct_bt_MCMC)
    %     end
    
    %     switch param.type_data
    %         case {'inc3D_Re3900_blocks_truncated', ...
    %                 'incompact3d_wake_episode3_cut_truncated', ...
    %                 'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_truncated'}
    % %             plot_bt_dB_MCMC_varying_error(param,zzz,zzz,...
    % %                 zzz, zzz, bt_forecast_deter,...
    % %                 zzz,bt_forecast_sto,zzz,bt_tot,struct_bt_MCMC)
    
    [ idx_min_error, idx_min_err_tot] = ...
        plot_bt_dB_MCMC_varying_error(param,zzz,zzz,...
        bt_forecast_sto, zzz, bt_forecast_deter,...
        bt_forecast_MEV,bt_forecast_MEV_noise,struct_bt_MEV_noise,...
        bt_tot,struct_bt_MCMC,bt_MCMC);
%         bt_forecast_MEV,bt_forecast_sto,zzz,bt_tot,struct_bt_MCMC,bt_MCMC)
    save(file_res_2nd_res,'idx_min_error','idx_min_err_tot','-append')
    %         otherwise
    %             %     plot_bt_dB_MCMC(param,zzz,zzz,...
    %             %     plot_bt_dB_MCMC_more_subplots(param,zzz,zzz,...
    %             plot_bt_MCMC_more_subplots(param,zzz,zzz,...
    %                 zzz, zzz, bt_forecast_deter,...
    %                 zzz,bt_forecast_sto,zzz,bt_tot,struct_bt_MCMC,bt_MCMC)
    %     end
end
% if plot_bts
%     plot_bt5(param,bt_forecast_sto,bt_forecast_deter,bt_tot)
% end
toc;tic
disp('plot done');

