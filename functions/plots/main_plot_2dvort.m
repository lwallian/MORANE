function main_plot_2dvort(type_data,nb_modes,...
    threshold,no_subampl_in_forecast,reconstruction,adv_corrected,modal_dt,...
    svd_pchol,eq_proj_div_free,...
    data_assimilation,coef_bruit_obs,param_obs)
% Load simulation results, estimate modal time step by Shanon
% and compare it with modal Eddy Viscosity ROM and
% tuned version of the loaded results
%

clear param bt_forecast_sto bt_forecast_deter bt_tot
tic


global choice_n_subsample;
global stochastic_integration;
global estim_rmv_fv;
global correlated_model;

param.svd_pchol = svd_pchol;
param.eq_proj_div_free = eq_proj_div_free;

param.DA.beta_2 = 1; % beta_2 is the parameter that controls the  noise in the initialization of the filter
param.DA.init_centred_on_ref = false;  % If True, the initial condition is centered on the reference initial condiion


%% Parameters choice

% Plots to do
plot_deterministic=true; % deterministic POD-Galerkin
plot_EV=true; % estimated Eddy Visocvity
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
test_fct='b'; % 'b' is better than db

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
folder_data_PIV = [ pwd '/data_PIV/' ];
cd(current_pwd);
% folder_results = ['/Users/Resseguier/Documents/MATLAB/POD/all/resultats/current_results/'];
% folder_data = '/Users/Resseguier/Documents/MATLAB/POD/data/';
% folder_results = ['/Users/vressegu/Documents/matlab/POD-NS_Stochastique/current_used/' ...
%     'all/resultats/current_results/'];
%     param.folder_results =  [ pwd '/resultats/current_results/'];
param_ref2.folder_results=folder_results;
param_ref2.folder_data =folder_data ;
param_ref2.folder_data_PIV =folder_data_PIV ;

if nargin > 0
    plot_each_mode=false;
end

modal_dt_ref = modal_dt;
folder_results_ref = folder_results;
folder_data_ref = folder_data;
folder_data_PIV_ref = folder_data_PIV;

%% Get data

% a_t='_a_cst_';
%
% file_res=[ folder_results '2ndresult_' type_data '_' num2str(nb_modes) '_modes_' ...
%     a_t '_decor_by_subsampl_bt_decor_choice_auto_shanon_threshold_' ...
%     num2str(threshold) ...
%     'fct_test_' test_fct ];
%
% file_res=[file_res '_fullsto'];
% % if modal_dt
% %     file_res=[file_res '_modal_dt'];
% % end
% if modal_dt == 1
%     file_res=[file_res '_modal_dt'];
% elseif modal_dt == 2
%     file_res=[file_res '_real_dt'];
% end
% if ~ adv_corrected
%     file_res=[file_res '_no_correct_drift'];
% end
% if no_subampl_in_forecast
%     file_res=[file_res '_no_subampl_in_forecast'];
% end
% if reconstruction
%     file_res=[file_res '_reconstruction'];
% end
%
% % file_res=[file_res '_fullsto'];
% file_res=[file_res '.mat'];
% load(file_res)

param.nb_modes = nb_modes;
param.decor_by_subsampl.spectrum_threshold = threshold;
param.adv_corrected = adv_corrected;

param.decor_by_subsampl.no_subampl_in_forecast = no_subampl_in_forecast;
param.decor_by_subsampl.test_fct=test_fct;

param.folder_data = folder_data;
param.folder_data_PIV = folder_data_PIV;
param.folder_results = folder_results;
param.type_data = type_data;
param.nb_modes = nb_modes;
% param.big_data=big_data;
% param.plot_bts=plot_bts;
param.coef_correctif_estim=coef_correctif_estim;

param.a_time_dependant = false;
param.decor_by_subsampl.bool = true;
param.decor_by_subsampl.meth = 'bt_decor';
% param.decor_by_subsampl.choice_n_subsample = 'auto_shanon';
param.data_assimilation=data_assimilation;

switch data_assimilation
    case 0
        param = fct_name_2nd_result_new(param,modal_dt,reconstruction);
        %         param = fct_name_2nd_result(param,modal_dt,reconstruction);
        load(param.name_file_2nd_result);
    case 1
        if param.nb_modes == 6
            param.name_file_2nd_result = [ param.folder_results ...
                'data_to_valentin_beta_noise_' num2str( coef_bruit_obs ) ...
                'Reynolds100.mat' ];
            load(param.name_file_2nd_result);
            n_simu = double(n_simu);
            param.DA.bool = true;
            param.DA.coef_bruit_obs = coef_bruit_obs;
            param.N_test = size(bt_tot,1);
            bt_forecast_EV = bt_forecast_EV(1:n_simu:end,:);
            bt_MCMC = bt_MCMC(1:n_simu:end,:);
            param.dt = dt * n_simu;
            param.d = double(param.d);
            param.MX = double(param.MX);
            param.M = double(param.M);
            param.N_tot = double(param.N_tot);
            param.decor_by_subsampl.n_subsampl_decor = ...
                double(param.decor_by_subsampl.n_subsampl_decor);
            param.param.N_particules = double(param.N_particules);
            %         index_of_filtering;
            param.DA.index_of_filtering = [1 (index_of_filtering+1)/n_simu];
        else
            error('Not available')
        end
    case 2
        n_simu = 100
        param.n_simu = n_simu;
        current_pwd = pwd; cd ..
        param.folder_results = [ pwd '/3rdresult/'];
        cd(current_pwd); clear current_pwd
        [param,param_obs] = fct_name_3rd_result_new(param,param_obs,modal_dt);
        
        name_simu = param.name_simu;
        load(param.name_file_3rd_result);
        param.name_simu = name_simu;                
        param.svd_pchol = svd_pchol;
        
        param.DA.bool = true;
        param.DA.coef_bruit_obs = nan;
        param.DA.index_of_filtering = index_pf+1;
%         param.DA.index_of_filtering = [1 (index_of_filtering+1)/n_simu];
        
%         param.dt = param.dt*N_simu;

        [time,i_time,~] = unique(time,'last');
        bt_MCMC = bt_MCMC(i_time,:,:);
        bt_forecast_EV = bt_forecast_EV(i_time,:,:);        

        [time_inter,i_time,i_time_bt_tot] = intersect(time,time_bt_tot);
        time = time_inter;
        time_bt_tot = time_inter;
        
        dt_ini = param.dt * n_simu ...
                / double(param.decor_by_subsampl.n_subsampl_decor);
        param.dt = time_inter(2)-time_inter(1);
        param.decor_by_subsampl.n_subsampl_decor = ...
            param.dt/dt_ini;
        
        bt_tot = bt_tot(i_time_bt_tot,:);
        bt_MCMC = bt_MCMC(i_time,:,:);
        bt_forecast_EV = bt_forecast_EV(i_time,:,:);
        param.N_test = size(bt_tot,1);
        
        bt_MCMC = mean(bt_MCMC,3);
        bt_forecast_EV = mean(bt_forecast_EV,3);
        
%         time_max = max([time time_bt_tot]);
%         time(time>time_max)=[];
%         time_bt_tot(time_bt_tot>time_max)=[];
%         idx_time_keep = 
        
        %         file_plots = [ type_data '_' str(nb_modes) '_modes_'  ...
        %          param.decor_by_subsampl.choice_n_subsample ];
        %         if strcmp(choice_n_subsample, 'auto_shanon')
        %             file_plots = [file_plots  '_threshold_' ...
        %                 str(param.decor_by_subsampl.spectrum_threshold)];
        %         end
        %         file_plots = [file_plots  param.decor_by_subsampl.test_fct  '/'];
        %         if modal_dt
        %              file_plots = [file_plots  '_modal_dt'];
        %         end
        %         if ~ adv_corrected
        %              file_plots = [file_plots  '_no_correct_drift'];
        %         end
        %
        %
        %         file_plots = [file_plots '_integ_' stochastic_integration];
        %         if correlated_model
        %             file_plots = [file_plots '_correlated'];
        %         end
        %         if estim_rmv_fv
        %             file_plots=[file_plots '_estim_rmv_fv'];
        %         end
        %         if param.svd_pchol
        %             file_plots=[file_plots '_svd_pchol'];
        %         end
        %         if param.eq_proj_div_free == 2
        %             file_plots = [file_plots '_DFSPN'];
        %         end
        % %         file_plots = [file_plots  '_integ_Ito'];
        %         if svd_pchol
        %              file_plots = [file_plots  '_svd_pchol'];
        %         end
        %         file_plots = [file_plots  '/'  param_obs.assimilate  ...
        %             '/_DADuration_'  str(param_obs.SECONDS_OF_SIMU)  '_'];
        %         switch param_obs.assimilate
        %             case 'real_data'
        %                 switch type_data
        %                     case 'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_truncated'
        %                         param_obs.dt_PIV = 0.080833;
        %                     case 'DNS100_inc3d_2D_2018_11_16_blocks_truncated'
        %                         param_obs.dt_PIV = 0.05625;
        %                 end
        %             case 'fake_real_data'
        %                 if strcmp(choice_n_subsample, 'auto_shanon')
        %                     switch type_data
        %                         case 'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_truncated'
        %                             param_obs.dt_PIV = 0.25;
        %                         case 'DNS100_inc3d_2D_2018_11_16_blocks_truncated'
        %                             param_obs.dt_PIV = 0.05;
        %                     end
        %                 else
        %                     error('unknown')
        %                 end
        %         end
        %         if param_obs.sub_sampling_PIV_data_temporaly
        %              param_obs.factor_of_PIV_time_subsampling = int(param_obs.assimilation_period/param_obs.dt_PIV)
        %              file_plots = [file_plots  'ObsSubt_'  str(param_obs.factor_of_PIV_time_subsampling) '_'];
        %         end
        %         if param_obs.mask_obs
        %              file_plots = [file_plots  'ObsMaskyy_sub_'  str(param_obs.subsampling_PIV_grid_factor) ...
        %                 '_from_'  str(param_obs.x0_index)  '_to_ ' ...
        %                 str(param_obs.nbPoints_x+param_obs.x0_index*param_obs.subsampling_PIV_grid_factor) ...
        %                 '_from_'  str(param_obs.y0_index)  '_to_ ' ...
        %                 str(param_obs.nbPoints_y+param_obs.y0_index*param_obs.subsampling_PIV_grid_factor)  '_'];
        %         else
        %             param_obs.x0_index =1;
        %             param_obs.y0_index =1;
        %             param_obs.subsampling_PIV_grid_factor = 1;
        %             file_plots = [file_plots  'no_mask_'];
        %         end
        %         if init_centred_on_ref
        %              file_plots = [file_plots  'initOnRef_'];
        %         end
        %         file_plots = [file_plots  'beta_2_'  str(beta_2)]
        %         param.name_file_2nd_result = [ param.folder_results ...
        %             file_plots 'chronos.mat' ];
        %         load(param.name_file_2nd_result);
    otherwise
        error('Unknown DA type')
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
folder_data_PIV = folder_data_PIV_ref;
param.folder_data_PIV = folder_data_PIV;
modal_dt = modal_dt_ref;
modal_dt

param.decor_by_subsampl.test_fct=test_fct;

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
param.data_assimilation=data_assimilation;

% if strcmp(param.type_data, 'inc3D_Re3900_blocks')
%     param.N_test = ceil(10*5/param.dt);
%     warning('simulation on only 10 periods')
% end

param.folder_results=param_ref2.folder_results;

% param.N_particules=param_ref2.N_particules;
% n_simu=param_ref2.n_simu;


% struct_bt_MCMC.tot.one_realiz = bt_MCMC(:,:,4);
struct_bt_MCMC.tot.one_realiz = bt_MCMC(:,:,1);



%% Eddy viscosity solutions
% if plot_EV
%     param.plot.plot_EV= plot_EV;
%     file_EV=[ param.folder_results 'EV_result_' param.type_data ...
%         '_' num2str(param.nb_modes) '_modes'];
%     file_EV=[file_EV '.mat'];
%     load(file_EV,'param_deter',...
%         'bt_forecast_MEV','bt_forecast_EV','bt_forecast_NLMEV');
%     %     load(file_EV,'param_deter',...
%     %         'bt_forecast_deter',...
%     %         'bt_forecast_MEV','bt_forecast_EV','bt_forecast_NLMEV');
% %     bt_forecast_MEV = bt_forecast_EV;
% %     if modal_dt ~= 1
% %         bt_forecast_MEV = bt_forecast_EV;
% %     end
% %      clear bt_forecast_EV
%     bt_forecast_MEV = ...
%         bt_forecast_MEV(1:param.decor_by_subsampl.n_subsampl_decor:end,:);
%     bt_forecast_EV = ...
%         bt_forecast_EV(1:param.decor_by_subsampl.n_subsampl_decor:end,:);
% end

%% Eddy viscosity solutions
% if plot_EV
if plot_EV && (data_assimilation == 0)
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
    %     clear bt_forecast_EV
    bt_forecast_EV = ...
        bt_forecast_EV(1:param.decor_by_subsampl.n_subsampl_decor:end,:);
    bt_forecast_MEV = ...
        bt_forecast_MEV(1:param.decor_by_subsampl.n_subsampl_decor:end,:);
end

%% Plots
% % plot_isoQ(param,'ref', nan, reconstruction);
% plot_isoQ(param,'mean', modal_dt,reconstruction);
% keyboard;

%% Computation of 2dvort criterion
% param = ref_2dvort(param,reconstruction);
switch data_assimilation
    case 0
        param = ref_2dvort(param,reconstruction);
        % if ~ param.DA.bool
        bt_err_min=nan(size(bt_MCMC(1:param.N_test,:,idx_min_err_tot)));
        for t=1:param.N_test
            bt_err_min(t,:,:) = bt_MCMC(t,:,idx_min_error(t));
        end
        param = reconstruction_2dvort( ...
            param,bt_err_min,'min_local',...
            modal_dt,reconstruction);
        param = reconstruction_2dvort( ...
            param,bt_MCMC(1:param.N_test,:,idx_min_err_tot),'min_tot',...
            modal_dt,reconstruction);
        param = reconstruction_2dvort( ...
            param,struct_bt_MCMC.tot.mean(1:param.N_test,:),'mean',...
            modal_dt,reconstruction);
        param = reconstruction_2dvort( ...
            param,struct_bt_MCMC.tot.one_realiz(1:param.N_test,:),'1_realiz',...
            modal_dt,reconstruction);
        param = reconstruction_2dvort( ...
            param,bt_forecast_EV(1:param.N_test,:),'EV',...
            modal_dt,reconstruction);
        param = reconstruction_2dvort( ...
            param,bt_forecast_MEV(1:param.N_test,:),'MEV',...
            modal_dt,reconstruction);
        param = reconstruction_2dvort( ...
            param,bt_forecast_deter(1:param.N_test,:),'deter',...
            modal_dt,reconstruction);
    case 1
        param = ref_2dvort(param,reconstruction);
        param = reconstruction_2dvort( ...
            param,bt_forecast_EV(1:param.N_test,:),['3DVar_EV', ...
            num2str(param.DA.coef_bruit_obs) ],...
            modal_dt,reconstruction);
        param = reconstruction_2dvort( ...
            param,bt_tot(1:param.N_test,:),'PODROM_Opt',...
            modal_dt,reconstruction);
        param = reconstruction_2dvort( ...
            param,bt_MCMC(1:param.N_test,:),[ 'PF_LU' ...
            num2str(param.DA.coef_bruit_obs) ],...
            modal_dt,reconstruction);
        %     param = reconstruction_2dvort( ...
        %         param,bt_MCMC(1:param.N_test,:),[ 'DA_' ...
        %         num2str(param.DA.coef_bruit_obs) ],...
        %         modal_dt,reconstruction);
    case 2
        param.param_obs = param_obs;
        param.param_obs.no_noise = true;
        param = ref_2dvort(param,reconstruction);
        param.param_obs.no_noise = false;
        param = ref_2dvort(param,reconstruction);
%         param = reconstruction_2dvort( ...
%             param,bt_forecast_EV(1:param.N_test,:),['PF_EV', ...
%             num2str(param.DA.coef_bruit_obs) ],...
%             modal_dt,reconstruction,param_obs);
        param = reconstruction_2dvort( ...
            param,bt_tot(1:param.N_test,:),'PODROM_Opt',...
            modal_dt,reconstruction);
        param = reconstruction_2dvort( ...
            param,bt_MCMC(1:param.N_test,:),[ 'PF_LU' ...
            '_' param.name_simu ],...
            modal_dt,reconstruction);
        %     param = reconstruction_2dvort( ...
        %         param,bt_MCMC(1:param.N_test,:),[ 'DA_' ...
        %         num2str(param.DA.coef_bruit_obs) ],...
        %         modal_dt,reconstruction);
end
% param = reconstruction_2dvort( ...
%     param,bt_tot(1:param.N_test,:),'PODROM_Opt',...
%     modal_dt,reconstruction);


%% Plots
% plot_iso2dvort(param,'mean', modal_dt,reconstruction);


% param = reconstruction_velocity( ...
%     param,struct_bt_MCMC.tot.mean(1:param.N_test,:),'mean',...
%     modal_dt,reconstruction);
