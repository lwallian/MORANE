function main_full_sto_modal_dt(nb_modes,threshold)
% Load simulation results, estimate modal time step by Shanon
% and compare it with modal Eddy Viscosity ROM and
% tuned version of the loaded results
%

clear param bt_forecast_sto bt_forecast_deter bt_tot
tic

%% Parameters choice

% Plots to do
plot_deterministic=true; % deterministic POD-Galerkin
plot_EV=false; % estimated Eddy Visocvity
plot_tuned=false; % estimated corrective coefficients
plot_modal_dt=false; % different time step (infered by Shanon criterion) for the different modes

plot_each_mode=true;

% Threshold of the Chonos spectrum, used to choice the time step
% threshold=0.001; % 0.001 or 0.01 for LES 3900 (or inc3D 3900)
% threshold=0.005; % for LES 3900
threshold=0.0005; % for inc3D episode 3
% threshold=0.00014; % for inc3D 3900
% threshold=0.000135; % for inc3D 3900

% Number of particle for the MCMC simulation
param_ref.N_particules=1000;
% param_ref.N_particules=2;
% param.N_particules=2;
% % param.N_particules=1000;

% Rate of increase of the time step to simulate accurately the SDE
param_ref.n_simu = 100;
% param_ref.n_simu = 1;
% n_simu = 1;
% % n_simu = 100;

% On which function the Shanon ctriterion is used
test_fct='b'; % 'b' is better than db

% Learning duration
% period_estim=3;
% % p_estim=13;
% % N_estim=842;
% coef_correctif_estim.learning_time='N_estim'; % 'quarter' or 'all'
% % coef_correctif_estim.learning_time='quarter'; % 'quarter' or 'all'

% Type of data

% These 3D data give good results
% They are saved in only one file
% (~ 250 time step)
% type_data = 'LES_3D_tot_sub_sample_blurred';
type_data = 'incompact3d_wake_episode3_cut';
% type_data = 'inc3D_Re3900_blocks';
%     type_data = 'incompact3D_noisy2D_40dt_subsampl'; 

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

folder_results = ['/Users/Resseguier/Documents/MATLAB/POD/all/resultats/current_results/'];
folder_data = '/Users/Resseguier/Documents/MATLAB/POD/data/';
% folder_results = ['/Users/vressegu/Documents/matlab/POD-NS_Stochastique/current_used/' ...
%     'all/resultats/current_results/'];
%     param.folder_results =  [ pwd '/resultats/current_results/'];
param_ref.folder_results=folder_results;
param_ref.folder_data =folder_data ;

if nargin > 0
    plot_each_mode=false;
end

%% Get data

a_t='_a_cst_';
% 
% file_res=[ folder_results '2ndresult_' type_data '_' num2str(nb_modes) '_modes_' ...
file_res=[ folder_results '1stresult_' type_data '_' num2str(nb_modes) '_modes_' ...
    a_t '_decor_by_subsampl_bt_decor_choice_auto_shanon_threshold_' ...
    num2str(threshold) ...
    'fct_test_' test_fct ];
if exist('period_estim','var')
    file_res=[file_res '_p_estim_' num2str(period_estim)];
end
file_res=[ file_res '.mat'];
load(file_res)

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

% %% Parameters of the ODE of the b(t)
% tot = struct('I',I_sto,'L',L_sto,'C',C_sto);
% 
% I_sto = I_sto - I_deter ;
% L_sto = L_sto - L_deter;
% C_sto = C_sto - C_deter;
% 
% deter = struct('I',I_deter,'L',L_deter,'C',C_deter);
% sto = struct('I',I_sto,'L',L_sto,'C',C_sto);
% ILC=struct('deter',deter,'sto',sto,'tot',tot);
% % ILC=struct('deter',deter,'sto',sto);
% 
% ILC_a_cst=ILC;
% % bt_sans_coef_a_cst = bt_forecast_sto;
% 
% %%
% param.decor_by_subsampl.test_fct=test_fct;
% 
folder_data = param_ref.folder_data;
folder_results = param_ref.folder_results;
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
% 
% %% Choice the learning duration
% if ~isfield(param.coef_correctif_estim,'learning_time')
%     param.coef_correctif_estim.learning_time='quarter';
% end
% switch param.coef_correctif_estim.learning_time
%     case 'all'
%         param.N_learn_coef_a=size(bt_tot,1);
%     case 'quarter'
%         param.N_learn_coef_a=ceil((size(bt_tot,1)-2)/4);
%     case 'N_estim'
%         param.N_learn_coef_a=param.N_estim;
%     otherwise
%         error('unknown type of duration');
% end
% 
% %% Learning corrective coefficients
% % 
% % [coef_beta, ILC_beta] = estim_vector_mat_beta(bt_tot,ILC,param);
% % param.coef_correctif_estim.type_estim='scalar';
% % [coef_scalar, ILC_scalar] = estim_vector_mat_beta(bt_tot,ILC,param);
% % [MEV, ILC] = estim_modal_eddy_viscosity(bt_tot,ILC,param);
% % [NLMEV, ILC] = estim_modal_non_lin_eddy_viscosity(bt_tot,ILC,param);
% 
% %% Time integration of the reconstructed Chronos b(t)
% 
% param.folder_results=param_ref.folder_results;
% 
% param.N_particules=param_ref.N_particules;
% n_simu=param_ref.n_simu;
% 
% bt_tot=bt_tot(1:(param.N_test+1),:); % reference Chronos
% bt_tronc=bt_tot(1,:); % Initial condition
% 
% param.dt = param.dt/n_simu;
% param.N_test=param.N_test*n_simu;

%% Load 2nd results, especially I, L, C and the reconstructed Chronos
if param.decor_by_subsampl.bool
    if strcmp(dependance_on_time_of_a,'a_t')
        char_filter = [ '_on_' param.type_filter_a ];
    else
        char_filter = [];
    end
    file_save=[ param.folder_results '2ndresult_' param.type_data '_' num2str(param.nb_modes) '_modes_' ...
        dependance_on_time_of_a char_filter ...
        '_decor_by_subsampl_' num2str(param.decor_by_subsampl.meth) ...
        '_choice_' num2str(param.decor_by_subsampl.choice_n_subsample)  ...
        '_threshold_' num2str(param.decor_by_subsampl.spectrum_threshold) ...
        'fct_test_' param.decor_by_subsampl.test_fct ];
else
    file_save=[ param.folder_results '2ndresult_' param.type_data '_' num2str(param.nb_modes) '_modes_' ...
        dependance_on_time_of_a ];
end
file_save=[file_save '_fullsto'];
file_save=[file_save '.mat'];
load(file_save);
% save(file_save);
clear C_deter C_sto L_deter L_sto I_deter I_sto
% if param.big_data
    toc;tic
    disp('2nd result saved');
% end

%% Plots of the reconstructed Chronos
if plot_bts
    param.folder_data =param_ref.folder_data ;
    
    param.plot.plot_deter=plot_deterministic;
    param.plot.plot_EV=plot_EV;
    param.plot.plot_tuned=plot_tuned;
    param.plot_modal_dt = plot_modal_dt;
    
    plot_bt_dB_MCMC(param,bt_tot,bt_tot,...
            bt_tot, bt_tot, bt_forecast_deter,...
            bt_tot,bt_forecast_sto,bt_forecast_sto,bt_tot,struct_bt_MCMC)
    
    if plot_each_mode
        plot_bt_MCMC(param,bt_tot,bt_tot,...
            bt_tot, bt_tot, bt_forecast_deter,...
            bt_tot,bt_forecast_sto,bt_forecast_sto,bt_tot,struct_bt_MCMC)
    end
end
% if plot_bts
%     plot_bt5(param,bt_forecast_sto,bt_forecast_deter,bt_tot)
% end
    toc;tic
    disp('plot done');
    
