function param = main_modal_dt(nb_modes,type_data,threshold,igrida,coef_correctif_estim)
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

if nargin > 0
    plot_each_mode=false;
end

%% Get data
coef_correctif_estim_ref=coef_correctif_estim;

a_t='_a_time_dependant_';

% file_res=[ folder_results '2ndresult_' type_data '_' num2str(nb_modes) '_modes_' ...
file_res=[ folder_results '1stresult_' type_data '_' num2str(nb_modes) '_modes_' ...
    a_t '_decor_by_subsampl_bt_decor_choice_auto_shanon_threshold_' ...
    num2str(threshold)  ...
    'fct_test_' test_fct ];
if exist('period_estim','var')
    file_res=[file_res '_p_estim_' num2str(period_estim)]; 
end
file_res=[ file_res '.mat'];
load(file_res);
% load([ folder_results '2ndresult_' type_data '_' num2str(nb_modes) '_modes_' ...
%     a_t '_decor_by_subsampl_bt_decor_choice_auto_shanon_threshold_' ...
%     num2str(threshold)  ...
%     'fct_test_' test_fct '.mat']);

%% Parameters of the ODE of the b(t)
tot = struct('I',I_sto,'L',L_sto,'C',C_sto);

I_sto = I_sto - I_deter ;
L_sto = L_sto - L_deter;
C_sto = C_sto - C_deter;

deter = struct('I',I_deter,'L',L_deter,'C',C_deter);
sto = struct('I',I_sto,'L',L_sto,'C',C_sto);
ILC=struct('deter',deter,'sto',sto,'tot',tot);
% ILC=struct('deter',deter,'sto',sto);

ILC_a_NC=ILC;
% bt_sans_coef_a_NC = bt_forecast_sto;

%%
a_t='_a_cst_';

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
% load([ folder_results '2ndresult_' type_data '_' num2str(nb_modes) '_modes_' ...
%     a_t '_decor_by_subsampl_bt_decor_choice_auto_shanon_threshold_' ...
%     num2str(threshold) ...
%     'fct_test_' test_fct '.mat']);

%% Parameters of the ODE of the b(t)
tot = struct('I',I_sto,'L',L_sto,'C',C_sto);

I_sto = I_sto - I_deter ;
L_sto = L_sto - L_deter;
C_sto = C_sto - C_deter;

deter = struct('I',I_deter,'L',L_deter,'C',C_deter);
sto = struct('I',I_sto,'L',L_sto,'C',C_sto);
ILC=struct('deter',deter,'sto',sto,'tot',tot);
% ILC=struct('deter',deter,'sto',sto);

ILC_a_cst=ILC;
% bt_sans_coef_a_cst = bt_forecast_sto;

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

%% Choice of modal time step

[rate_dt, ILC_a_cst] = fct_cut_frequency_2(bt_tot,ILC_a_cst,param);
[rate_dt, ILC_a_NC] = fct_cut_frequency_2(bt_tot,ILC_a_NC,param);

%% Choice the learning duration
if ~isfield(param.coef_correctif_estim,'learning_time')
    param.coef_correctif_estim.learning_time='quarter';
end
switch param.coef_correctif_estim.learning_time
    case 'all'
        param.N_learn_coef_a=size(bt_tot,1);
    case 'quarter'
            param.N_learn_coef_a=ceil((size(bt_tot,1)-2)/4);
    case 'N_estim'
            param.N_learn_coef_a=param.N_estim;
    otherwise
        error('unknown type of duration');
end

%% Learning corrective coefficients
if ~isfield(param,'C_deter_residu')
    [~,~,~,~,param_C_residu] ...
        = fct_all_file_save_1st_result(param);
    param.C_deter_residu = param_C_residu.C_deter_residu;
    clear param_C_residu
end

[coef_beta, ILC_beta] = estim_vector_mat_beta(bt_tot,ILC,param);
param.coef_correctif_estim.type_estim='scalar';
[coef_scalar, ILC_scalar] = estim_vector_mat_beta(bt_tot,ILC,param);
[EV, ILC] = estim_eddy_viscosity(bt_tot,ILC,param);
[MEV, ILC] = estim_modal_eddy_viscosity(bt_tot,ILC,param);
[NLMEV, ILC] = estim_modal_non_lin_eddy_viscosity(bt_tot,ILC,param);

%% Time integration of the b(t)

warning('local change of the time step')
n_mult_dt = 10;
if param.decor_by_subsampl.n_subsampl_decor >= 5
    n_mult_dt = n_mult_dt * param.decor_by_subsampl.n_subsampl_decor;
end
%n_mult_dt = 1;


% warning('N_test modified');
% param.N_test=ceil(param.N_test/6);
% % param.N_test=ceil(param.N_test/12);

bt_init=bt_tot(1:(end-param.N_test),:);
% bt_tot=bt_tot(1:(param.N_test+1),:);
% bt_sans_coef_a_cst=bt_sans_coef_a_cst(1:(param.N_test+1),:);
% bt_sans_coef_a_NC=bt_sans_coef_a_NC(1:(param.N_test+1),:);


% warning('N_test modified');
% param.N_test=ceil(param.N_test/6);
% % param.N_test=ceil(param.N_test/12);

param.N_tot_old=param.N_tot;
param.N_tot=param.N_test+1;
bt_tot=bt_tot(1:param.N_tot,:);
bt_init=bt_tot(1,:);


param.dt = param.dt /n_mult_dt;
param.N_test = param.N_test *n_mult_dt;

N_test=param.N_test;
dt_tot=param.dt;
bt_forecast_deter=bt_init;
for l = 1:N_test
    bt_forecast_deter= [bt_forecast_deter; ...
        evol_forward_bt_RK4(I_deter,L_deter,C_deter, dt_tot, bt_forecast_deter)];
end
bt_forecast_MEV=bt_init;
for l = 1:N_test
    bt_forecast_MEV= [bt_forecast_MEV; ...
        evol_forward_bt_RK4(ILC.MEV.I,ILC.MEV.L,ILC.MEV.C, dt_tot, bt_forecast_MEV)];
end
bt_forecast_EV=bt_init;
for l = 1:N_test
    bt_forecast_EV= [bt_forecast_EV; ...
        evol_forward_bt_RK4(ILC.EV.I,ILC.EV.L,ILC.EV.C, dt_tot, bt_forecast_EV)];
end
bt_forecast_NLMEV=bt_init;
for l = 1:N_test
    bt_forecast_NLMEV= [bt_forecast_NLMEV; ...
        evol_forward_bt_NLMEV_RK4(ILC.NLMEV, dt_tot, bt_forecast_NLMEV)];
end
bt_forecast_sto_scalar=bt_init;
for l = 1:N_test
    bt_forecast_sto_scalar = [bt_forecast_sto_scalar; ...
        evol_forward_bt_RK4(ILC_scalar.sto.I,ILC_scalar.sto.L,ILC_scalar.sto.C, ...
        dt_tot, bt_forecast_sto_scalar) ];
end
bt_forecast_sto_beta=bt_init;
for l = 1:N_test
    bt_forecast_sto_beta = [bt_forecast_sto_beta; ...
        evol_forward_bt_RK4(ILC_beta.sto.I,ILC_beta.sto.L,ILC_beta.sto.C, ...
        dt_tot, bt_forecast_sto_beta) ];
end
bt_forecast_sto_a_cst_modal_dt=bt_init;
for l = 1:N_test
    bt_forecast_sto_a_cst_modal_dt = [bt_forecast_sto_a_cst_modal_dt; ...
        evol_forward_bt_RK4(ILC_a_cst.modal_dt.I,ILC_a_cst.modal_dt.L,ILC_a_cst.modal_dt.C, ...
        dt_tot, bt_forecast_sto_a_cst_modal_dt) ];
end
bt_forecast_sto_a_NC_modal_dt=bt_init;
% bt_forecast_sto_a_NC_modal_dt=bt_tot(1,:);
for l = 1:N_test
    bt_forecast_sto_a_NC_modal_dt = [bt_forecast_sto_a_NC_modal_dt; ...
        evol_forward_bt_RK4(ILC_a_NC.modal_dt.I,ILC_a_NC.modal_dt.L,ILC_a_NC.modal_dt.C, ...
        dt_tot, bt_forecast_sto_a_NC_modal_dt) ];
end

% Reconstruction in the stochastic case
bt_sans_coef_a_cst=bt_init;
for l = 1:N_test
    bt_sans_coef_a_cst = [bt_sans_coef_a_cst; ...
        evol_forward_bt_RK4(ILC_a_cst.tot.I,ILC_a_cst.tot.L,ILC_a_cst.tot.C, ...
        dt_tot, bt_sans_coef_a_cst) ];
end
bt_sans_coef_a_NC=bt_init;
for l = 1:N_test
    bt_sans_coef_a_NC = [bt_sans_coef_a_NC; ...
        evol_forward_bt_RK4(ILC_a_NC.tot.I,ILC_a_NC.tot.L,ILC_a_NC.tot.C, ...
        dt_tot, bt_sans_coef_a_NC) ];
end


param.dt = param.dt *n_mult_dt;
param.N_test = param.N_test /n_mult_dt;
bt_forecast_deter = bt_forecast_deter(1:n_mult_dt:end,:);
bt_forecast_MEV = bt_forecast_MEV(1:n_mult_dt:end,:);
bt_forecast_EV = bt_forecast_EV(1:n_mult_dt:end,:);
bt_forecast_NLMEV = bt_forecast_NLMEV(1:n_mult_dt:end,:);
bt_forecast_sto_scalar = bt_forecast_sto_scalar(1:n_mult_dt:end,:);
bt_forecast_sto_beta = bt_forecast_sto_beta(1:n_mult_dt:end,:);
bt_forecast_sto_a_cst_modal_dt = bt_forecast_sto_a_cst_modal_dt(1:n_mult_dt:end,:);
bt_forecast_sto_a_NC_modal_dt = bt_forecast_sto_a_NC_modal_dt(1:n_mult_dt:end,:);
bt_sans_coef_a_cst = bt_sans_coef_a_cst(1:n_mult_dt:end,:);
bt_sans_coef_a_NC = bt_sans_coef_a_NC(1:n_mult_dt:end,:);



%% Save 3rd results, especially I, L, C and the reconstructed Chronos
% if param.a_time_dependant
%     dependance_on_time_of_a = '_a_time_dependant_';
% else
%     dependance_on_time_of_a = '_a_cst_';
% end
if param.decor_by_subsampl.bool
%     if strcmp(dependance_on_time_of_a,'a_t')
%         char_filter = [ '_on_' param.type_filter_a ];
%     else
%         char_filter = [];
%     end
        char_filter = [];
    %         save([ param.folder_results '2ndresult_' param.type_data '_' num2str(param.nb_modes) '_modes_' ...
    %             dependance_on_time_of_a char_filter ...
    %             '_decor_by_subsampl_' num2str(param.decor_by_subsampl.meth) ...
    %             '_choice_' num2str(param.decor_by_subsampl.choice_n_subsample)  ...
    %             '_threshold_' num2str(param.decor_by_subsampl.spectrum_threshold) ...
    %             'fct_test_' param.decor_by_subsampl.test_fct '.mat']);
    file_save = [ param.folder_results '3rdresult_' param.type_data '_' num2str(param.nb_modes) '_modes_' ...
        '_decor_by_subsampl_' num2str(param.decor_by_subsampl.meth) ...
        '_choice_' num2str(param.decor_by_subsampl.choice_n_subsample)  ...
        '_threshold_' num2str(param.decor_by_subsampl.spectrum_threshold) ...
        'fct_test_' param.decor_by_subsampl.test_fct];
else
    %     save([ param.folder_results '2ndresult_' param.type_data '_' num2str(param.nb_modes) '_modes_' ...
    %         dependance_on_time_of_a '.mat']);
    file_save=[ param.folder_results '3rdresult_' param.type_data '_' num2str(param.nb_modes) '_modes_' ...
        dependance_on_time_of_a];
end
if isfield(param,'N_estim')
    file_save=[file_save '_p_estim_' num2str(param.period_estim)];
end
file_save=[file_save '.mat'];
save(file_save);
clear C_deter C_sto L_deter L_sto I_deter I_sto
% if param.big_data
disp('3rd result saved');
% end

% Remove temporary files
if isfield(param,'folder_file_U_temp' )
    rmdir(param.folder_file_U_temp,'s');
end


%% Plots

param.plot.plot_deter=plot_deterministic;
param.plot.plot_EV=plot_EV;
param.plot.plot_tuned=plot_tuned;
param.plot_modal_dt = plot_modal_dt;

% plot_bt_dB(param,bt_forecast_sto_scalar,bt_forecast_sto_beta,...
%     bt_forecast_sto_a_cst_modal_dt, bt_forecast_sto_a_NC_modal_dt, bt_forecast_deter,...
%     bt_forecast_MEV,bt_sans_coef_a_cst,bt_sans_coef_a_NC,bt_tot)
plot_bt_dB(param,bt_forecast_EV,bt_forecast_sto_beta,...
    bt_forecast_sto_a_cst_modal_dt, bt_forecast_sto_a_NC_modal_dt, bt_forecast_deter,...
    bt_forecast_MEV,bt_sans_coef_a_cst,bt_sans_coef_a_NC,bt_tot)

plot_each_mode = true

% if plot_each_mode
%     plot_bt(param,bt_forecast_sto_scalar,bt_forecast_sto_beta,...
%         bt_forecast_sto_a_cst_modal_dt, bt_forecast_sto_a_NC_modal_dt, bt_forecast_deter,...
%         bt_forecast_MEV,bt_sans_coef_a_cst,bt_sans_coef_a_NC,bt_tot)
% end
