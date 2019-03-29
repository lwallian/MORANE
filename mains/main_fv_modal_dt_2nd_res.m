function param = main_fv_modal_dt_2nd_res(nb_modes,type_data,threshold,igrida,coef_correctif_estim)
% Load simulation results, estimate modal time step by Shanon 
% and compare it with modal Eddy Viscosity ROM and
% tuned version of the loaded results
%

clear param bt_forecast_sto bt_forecast_deter bt_tot

%% Parameters choice

% Plots to do
plot_deterministic=true; % deterministic POD-Galerkin
plot_EV=true; % estimated Eddy Visocvity
plot_tuned=false; % estimated corrective coefficients
plot_modal_dt=true; % different time step (infered by Shanon criterion) for the different modes

plot_each_mode=false;


% On which function the Shanon ctriterion is used
test_fct='b'; % 'b' is better than db

% Learning duration
% period_estim=3;
% % p_estim=13;
% % N_estim=842;
% coef_correctif_estim.learning_time='N_estim'; % 'quarter' or 'all'
% % coef_correctif_estim.learning_time='quarter'; % 'quarter' or 'all'


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

% folder_results = ['/Users/Resseguier/Documents/MATLAB/POD/all/resultats/current_results/'];
% folder_data = '/Users/Resseguier/Documents/MATLAB/POD/data/';
% % folder_results = ['/Users/vressegu/Documents/matlab/POD-NS_Stochastique/current_used/' ...
% %     'all/resultats/current_results/'];
% %     param.folder_results =  [ pwd '/resultats/current_results/'];

pwd_all = pwd;
folder_results =  [ pwd '/resultats/current_results/'];
cd ..
folder_data =  [ pwd '/data/'];
cd(pwd_all )
param_ref2.folder_results=folder_results;
param_ref2.folder_data =folder_data ;

if nargin > 0
    plot_each_mode=false;
end

%% Get data
coef_correctif_estim_ref=coef_correctif_estim;

 file_res=[ folder_results '3rdresult_' type_data '_' num2str(nb_modes) '_modes_' ...
    '_decor_by_subsampl_bt_decor_choice_auto_shanon_threshold_' ...
    num2str(threshold) ...
    'fct_test_' test_fct ];
if exist('period_estim','var')
    file_res=[file_res '_p_estim_' num2str(period_estim)]; 
end
file_res=[ file_res '.mat'];
load(file_res)

%%

if strcmp(param.type_data, 'inc3D_HRLESlong_Re3900_blocks')
%    time_resconstruction = 20;
    time_resconstruction = 15;
    param.N_test = ceil( time_resconstruction/param.dt)+1;
    
    bt_forecast_EV(param.N_test+1:end,:)=[];
    bt_forecast_sto_beta(param.N_test+1:end,:)=[];
    bt_forecast_sto_a_cst_modal_dt(param.N_test+1:end,:)=[];
    bt_forecast_sto_a_NC_modal_dt(param.N_test+1:end,:)=[];
    bt_forecast_deter(param.N_test+1:end,:)=[];
    bt_forecast_MEV(param.N_test+1:end,:)=[];
    bt_sans_coef_a_cst(param.N_test+1:end,:)=[];
    bt_sans_coef_a_NC(param.N_test+1:end,:)=[];
    bt_tot(param.N_test+1:end,:)=[];
end


%% Parameters of the ODE of the b(t)
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

%%



param.folder_results=param_ref2.folder_results;



%%
param.decor_by_subsampl.test_fct=test_fct;

% folder_data = '/Users/vressegu/Documents/matlab/POD-NS_Stochastique/current_used/data/';
% folder_results = ['/Users/vressegu/Documents/matlab/POD-NS_Stochastique/current_used/' ...
%         'all/resultats/current_results/'];
% %     param.folder_results =  [ pwd '/resultats/current_results/'];

pwd_all = pwd;
folder_results =  [ pwd '/resultats/current_results/'];
cd ..
folder_data =  [ pwd '/data/'];
cd(pwd_all )

big_data=false;
plot_bts=true;
coef_correctif_estim=coef_correctif_estim_ref;

param.folder_data = folder_data;
param.folder_results = folder_results;
param.big_data=big_data;
param.plot_bts=plot_bts;
param.coef_correctif_estim=coef_correctif_estim;


%% Plots

param.plot.plot_deter=plot_deterministic;
param.plot.plot_EV=plot_EV;
param.plot.plot_tuned=plot_tuned;
param.plot_modal_dt = plot_modal_dt;


if plot_each_mode
    plot_bt(param,bt_forecast_EV,bt_forecast_sto_beta,...
        bt_forecast_sto_a_cst_modal_dt, bt_forecast_sto_a_NC_modal_dt, bt_forecast_deter,...
        bt_forecast_MEV,bt_sans_coef_a_cst,bt_sans_coef_a_NC,bt_tot)
end

% plot_bt_dB(param,bt_forecast_EV,bt_forecast_sto_beta,...
%     bt_forecast_sto_a_cst_modal_dt, bt_forecast_sto_a_NC_modal_dt, bt_forecast_deter,...
%     bt_forecast_MEV,bt_sans_coef_a_cst,bt_sans_coef_a_NC,bt_tot)

plot_each_mode = true

