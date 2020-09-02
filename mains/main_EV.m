function main_EV(nb_modes,type_data,add_noise,rigorous_EV_noise_estim)
% param = main_EV(nb_modes,type_data,threshold,igrida,coef_correctif_estim)
% Load simulation results, estimate modal time step by Shanon 
% and compare it with modal Eddy Viscosity ROM and
% tuned version of the loaded results
%
no_subampl_in_forecast= true;
% no_subampl_in_forecast= false;
if strcmp(type_data((end-9):end),'_truncated')
    reconstruction = false;
else
    reconstruction = true;
end

global stochastic_integration;
stochastic_integration = 'Ito';
global estim_rmv_fv;
estim_rmv_fv = true;
svd_pchol=true;
eq_proj_div_free=2;
adv_corrected = true;
choice_n_subsample = 'htgen2';
% choice_n_subsample = 'htgen';
% warning('TO DO : change to htgen2');
threshold_effect_on_tau_corrected=false;
noise_type=0;
bug_sampling=false;
% bug_sampling=true;
% warning('TO DO ? : change to bug_sampling=false ??');

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
if nargin < 3
    add_noise = false;
    rigorous_EV_noise_estim = false;
end

% % On which function the Shanon ctriterion is used
% decor_by_subsampl.test_fct = 'b';

%% Parameters already chosen
% Do not modify the following lines

coef_correctif_estim.learn_coef_a=true; % true false
coef_correctif_estim.type_estim='vector_b'; % 'scalar' 'vector_z' 'vector_b' 'matrix'
coef_correctif_estim.beta_min=-inf; % -inf 0 1
coef_correctif_estim.nb_modes_used=eval('nb_modes'); % 2 eval('nb_modes') for learning the coefficient
coef_correctif_estim.learning_time='all'; % 'quarter' or 'all'
coef_correctif_estim_ref=coef_correctif_estim;

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


param_ref.folder_results=folder_results;
param_ref.folder_data =folder_data ;


%% Get data
% coef_correctif_estim_ref=coef_correctif_estim;
% 
% a_t='_a_time_dependant_';
% threshold = 1e-4;
% 
% % file_res=[ folder_results '2ndresult_' type_data '_' num2str(nb_modes) '_modes_' ...
% file_res=[ folder_results '1stresult_' type_data '_' num2str(nb_modes) '_modes_' ...
%     a_t '_decor_by_subsampl_bt_decor_choice_auto_shanon_threshold_' ...
%     num2str(threshold)  ...
%     'fct_test_' test_fct ];
% if exist('period_estim','var')
%     file_res=[file_res '_p_estim_' num2str(period_estim)]; 
% end
% file_res=[ file_res '.mat'];
% load(file_res);
% % load([ folder_results '2ndresult_' type_data '_' num2str(nb_modes) '_modes_' ...
% %     a_t '_decor_by_subsampl_bt_decor_choice_auto_shanon_threshold_' ...
% %     num2str(threshold)  ...
% %     'fct_test_' test_fct '.mat']);
% 
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
% ILC_a_NC=ILC;
% % bt_sans_coef_a_NC = bt_forecast_sto;

%%
%     a_t '_decor_by_subsampl_bt_decor_choice_auto_shanon_threshold_' ...
%     num2str(threshold) ...
param_ref.svd_pchol=svd_pchol;
param_ref.eq_proj_div_free=eq_proj_div_free;

param_ref.a_time_dependant = 0; % to account for the a_t
param_ref.decor_by_subsampl.bool = true; % we'll subsample

param_ref.decor_by_subsampl.choice_n_subsample = choice_n_subsample; % for testing
param_ref.decor_by_subsampl.bug_sampling = bug_sampling;
param_ref.decor_by_subsampl.threshold_effect_on_tau_corrected = ...
    threshold_effect_on_tau_corrected ;
param_ref.decor_by_subsampl.spectrum_threshold = nan;
param_ref.noise_type = noise_type;
param_ref.type_data = type_data;
param_ref.nb_modes = nb_modes;
param_ref.decor_by_subsampl.meth = 'bt_decor';

param_ref.adv_corrected = adv_corrected;

param_ref.decor_by_subsampl.test_fct = test_fct;

param_ref = fct_name_1st_result_new(param_ref);
load(param_ref.name_file_1st_result)

%% Load chronos
param_ROM = param;
load([ folder_results 'modes_bi_before_subsampling_' param.type_data ...
        '_nb_modes_' num2str(nb_modes) '.mat'],'bt','param');

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

% [rate_dt, ILC_a_cst] = fct_cut_frequency_2(bt_tot,ILC_a_cst,param);
% [rate_dt, ILC_a_NC] = fct_cut_frequency_2(bt_tot,ILC_a_NC,param);

%% Choice the learning duration
% if ~isfield(param.coef_correctif_estim,'learning_time')
%     param.coef_correctif_estim.learning_time='quarter';
% end
% switch param.coef_correctif_estim.learning_time
%     case 'all'
%         param.N_learn_coef_a=size(bt_tot,1);
%     case 'quarter'
%             param.N_learn_coef_a=ceil((size(bt_tot,1)-2)/4);
%     case 'N_estim'
%             param.N_learn_coef_a=param.N_estim;
%     otherwise
%         error('unknown type of duration');
% end

%% Learning corrective coefficients
if ~isfield(param,'C_deter_residu')
    [~,~,~,~,param_C_residu] ...
        = fct_all_file_save_1st_result(param_ROM);
%         = fct_all_file_save_1st_result(param);
    param.C_deter_residu = param_C_residu.C_deter_residu;
    clear param_C_residu
end

param.N_learn_coef_a = inf;
param.decor_by_subsampl.no_subampl_in_forecast = no_subampl_in_forecast;
param.nb_period_test = nan;
param.add_noise = add_noise;
param.rigorous_EV_noise_estim = rigorous_EV_noise_estim;

[coef_beta, ILC_beta] = estim_vector_mat_beta(bt_tot,ILC,param);
param.coef_correctif_estim.type_estim='scalar';
[coef_scalar, ILC_scalar] = estim_vector_mat_beta(bt_tot,ILC,param);
[EV, ILC] = estim_eddy_viscosity(bt_tot,ILC,param);
[MEV, ILC] = estim_modal_eddy_viscosity(bt_tot,ILC,param);
[NLMEV, ILC] = estim_modal_non_lin_eddy_viscosity(bt_tot,ILC,param);

%% Time integration of the b(t)
if param.add_noise
    n_simu = 100;
    param.N_particules = 100;
else
    n_simu = 10;
end
param.decor_by_subsampl.spectrum_threshold = 0;
param.decor_by_subsampl.n_subsampl_decor = 1;

warning('local change of the time step')
% n_simu = 100;
if param.decor_by_subsampl.n_subsampl_decor >= 5
    n_simu = n_simu * param.decor_by_subsampl.n_subsampl_decor;
end
%n_simu = 1;

%%

%% Do not temporally subsample, in order to prevent aliasing in the results
% BETA
if param.decor_by_subsampl.no_subampl_in_forecast
    param.dt = param.dt / param.decor_by_subsampl.n_subsampl_decor;
    param.N_test = param.N_test * param.decor_by_subsampl.n_subsampl_decor;
    param.N_tot = param.N_tot * param.decor_by_subsampl.n_subsampl_decor;
%     param.decor_by_subsampl.n_subsampl_decor = 1;
end

%% Creation of the test basis

n_subsampl_decor_ref = param.decor_by_subsampl.n_subsampl_decor;
param.decor_by_subsampl.n_subsampl_decor = 1;
[param,bt_tot,truncated_error2]=Chronos_test_basis(param,reconstruction);
if param.big_data
    toc;tic;
    disp('Creation of the test basis done');
end

%% Time integration of the reconstructed Chronos b(t)
param.folder_results=param_ref.folder_results;

% % warning('N_test modified');
% % param.N_test=ceil(param.N_test/6);
% % % param.N_test=ceil(param.N_test/12);
% 
% bt_init=bt_tot(1:(end-param.N_test),:);
% % bt_tot=bt_tot(1:(param.N_test+1),:);
% % bt_sans_coef_a_cst=bt_sans_coef_a_cst(1:(param.N_test+1),:);
% % bt_sans_coef_a_NC=bt_sans_coef_a_NC(1:(param.N_test+1),:);


% warning('N_test modified');
% param.N_test=ceil(param.N_test/6);
% % param.N_test=ceil(param.N_test/12);

% param.N_tot_old=param.N_tot;
% param.N_tot=param.N_test+1;
% bt_tot=bt_tot(1:param.N_tot,:);
param.N_tot = size(bt_tot,1);
param.N_test = param.N_tot-1;
bt_tot=bt_tot(1:(param.N_test+1),:); % reference Chronos
bt_init=bt_tot(1,:);


param.dt = param.dt /n_simu;
param.N_test = param.N_test *n_simu;
N_test=param.N_test;
dt_tot=param.dt;

bt_forecast_deter=nan([param.N_test+1 param.nb_modes]);
bt_forecast_deter(1,:)=bt_init;
for l = 1:N_test
    bt_forecast_deter(l+1,:) = RK4( bt_forecast_deter(l,:),...
        I_deter,L_deter,C_deter, dt_tot);
end
if param.add_noise
    bt_forecast_MEV=nan([param.N_test+1 param.nb_modes param.N_particules]);
    bt_forecast_MEV(1,:,:)=repmat(bt_init,[1 1 param.N_particules]);
%     bt_forecast_MEV=repmat(bt_forecast_MEV,[1 1 param.N_particules]);
else
    bt_forecast_MEV=nan([param.N_test+1 param.nb_modes ]);
    bt_forecast_MEV(1,:,:)=bt_init;    
end
for l = 1:N_test
    if ~param.add_noise
        bt_forecast_MEV(l+1,:,:) = RK4(bt_forecast_MEV(l,:,:),...
            ILC.MEV.I,ILC.MEV.L,ILC.MEV.C, dt_tot);
    else
        bt_forecast_MEV(l+1,:,:) = ...
            evol_forward_bt_MCMC(...
            ILC.MEV.I,ILC.MEV.L,ILC.MEV.C, ...
            ILC.MEV.pchol_cov_noises, param.dt, bt_forecast_MEV(l,:,:));
%         bt_forecast_MEV(l+1,:,:) = ...
%             evol_forward_bt_MCMC(...
%             ILC.MEV.I,ILC.MEV.L,ILC.MEV.C, ...
%             0, param.dt, bt_forecast_MEV(l,:,:));
%         bt_forecast_MEV(l+1,:,:)= bt_forecast_MEV(l+1,:,:) + ...
%             sqrt(dt_tot) * ILC.MEV.sigma_err ...
%             .* randn([1 param.nb_modes param.N_particules]);
% %             (1/sqrt(dt_tot)) * ILC.MEV.sigma_err ...
% %             .* randn([1 param.nb_modes param.N_particules]);
        iii_realization =  permute( any( ...
            isnan( bt_forecast_MEV(l+1,:,:) ) | isinf( bt_forecast_MEV(l+1,:,:) ) ...
            , 2) ,[3 1 2]); % N_particules
        if any(iii_realization)
            if all(iii_realization)
                warning('all realization of the simulation have blown up.')
                if l < param.N_test
                    bt_forecast_MEV((l+2):param.N_test,:,:) = ...
                        nan( param.N_test-l-1,param.nb_modes,param.N_particules);
                end
                break
            end
            nb_blown_up = sum(iii_realization);
            warning([ num2str(nb_blown_up) ...
                ' realizations have blown up and will be replaced.']);
            bt_forecast_MEV_good = bt_forecast_MEV(l+1,:, ~ iii_realization);
            rand_index =  randi( param.N_particules - nb_blown_up, nb_blown_up,1);
            bt_forecast_MEV(l+1,:, iii_realization) = bt_forecast_MEV_good(1,:, rand_index);
            clear bt_forecast_MEV_good rand_index nb_blown_up iii_realization
        end
    end
end
if param.add_noise
    bt_forecast_EV=nan([param.N_test+1 param.nb_modes param.N_particules]);
    bt_forecast_EV(1,:,:)=repmat(bt_init,[1 1 param.N_particules]);
%     bt_forecast_EV=repmat(bt_forecast_EV,[1 1 param.N_particules]);
else
    bt_forecast_EV=nan([param.N_test+1 param.nb_modes ]);
    bt_forecast_EV(1,:,:)=bt_init;    
end
for l = 1:N_test
    if ~param.add_noise
        bt_forecast_EV(l+1,:,:) = RK4(bt_forecast_EV(l,:,:),...
            ILC.EV.I,ILC.EV.L,ILC.EV.C, dt_tot);
    else
        bt_forecast_EV(l+1,:,:) = ...
            evol_forward_bt_MCMC(...
            ILC.EV.I,ILC.EV.L,ILC.EV.C, ...
            ILC.EV.pchol_cov_noises, param.dt, bt_forecast_EV(l,:,:));
%         bt_forecast_EV(l+1,:,:) = ...
%             evol_forward_bt_MCMC(...
%             ILC.EV.I,ILC.EV.L,ILC.EV.C, ...
%             0, param.dt, bt_forecast_EV(l,:,:));
%         bt_forecast_EV(l+1,:,:)= bt_forecast_EV(l+1,:,:) + ...
%             sqrt(dt_tot) * ILC.EV.sigma_err ...
%             .* randn([1 param.nb_modes param.N_particules]);
% %             (1/sqrt(dt_tot)) * ILC.EV.sigma_err ...
% %             .* randn([1 param.nb_modes param.N_particules]);
% %         bt_forecast_EV(end,:)= bt_forecast_EV(end,:) + ...
% %             (1/sqrt(dt_tot)) * ILC.EV.sigma_err .* randn(1,param.nb_modes);
        iii_realization =  permute( any( ...
            isnan( bt_forecast_EV(l+1,:,:) ) | isinf( bt_forecast_EV(l+1,:,:) ) ...
            , 2) ,[3 1 2]); % N_particules
        if any(iii_realization)
            if all(iii_realization)
                warning('all realization of the simulation have blown up.')
                if l < param.N_test
                    bt_forecast_EV((l+2):param.N_test,:,:) = ...
                        nan( param.N_test-l-1,param.nb_modes,param.N_particules);
                end
                break
            end
            nb_blown_up = sum(iii_realization);
            warning([ num2str(nb_blown_up) ...
                ' realizations have blown up and will be replaced.']);
            bt_forecast_EV_good = bt_forecast_EV(l+1,:, ~ iii_realization);
            rand_index =  randi( param.N_particules - nb_blown_up, nb_blown_up,1);
            bt_forecast_EV(l+1,:, iii_realization) = bt_forecast_EV_good(1,:, rand_index);
            clear bt_forecast_EV_good rand_index nb_blown_up iii_realization
        end
    end
end
bt_forecast_NLMEV=nan([param.N_test+1 param.nb_modes ]);
bt_forecast_NLMEV(1,:,:)=bt_init;
for l = 1:N_test
    bt_forecast_NLMEV(l+1,:,:) = RK4_NLMEV(bt_forecast_NLMEV(l,:,:),...
        ILC.NLMEV, dt_tot);
end
% bt_forecast_sto_scalar=bt_init;
% for l = 1:N_test
%     bt_forecast_sto_scalar = [bt_forecast_sto_scalar; ...
%         evol_forward_bt_RK4(ILC_scalar.sto.I,ILC_scalar.sto.L,ILC_scalar.sto.C, ...
%         dt_tot, bt_forecast_sto_scalar) ];
% end
% bt_forecast_sto_beta=bt_init;
% for l = 1:N_test
%     bt_forecast_sto_beta = [bt_forecast_sto_beta; ...
%         evol_forward_bt_RK4(ILC_beta.sto.I,ILC_beta.sto.L,ILC_beta.sto.C, ...
%         dt_tot, bt_forecast_sto_beta) ];
% end
% bt_forecast_sto_a_cst_modal_dt=bt_init;
% for l = 1:N_test
%     bt_forecast_sto_a_cst_modal_dt = [bt_forecast_sto_a_cst_modal_dt; ...
%         evol_forward_bt_RK4(ILC_a_cst.modal_dt.I,ILC_a_cst.modal_dt.L,ILC_a_cst.modal_dt.C, ...
%         dt_tot, bt_forecast_sto_a_cst_modal_dt) ];
% end
% bt_forecast_sto_a_NC_modal_dt=bt_init;
% % bt_forecast_sto_a_NC_modal_dt=bt_tot(1,:);
% for l = 1:N_test
%     bt_forecast_sto_a_NC_modal_dt = [bt_forecast_sto_a_NC_modal_dt; ...
%         evol_forward_bt_RK4(ILC_a_NC.modal_dt.I,ILC_a_NC.modal_dt.L,ILC_a_NC.modal_dt.C, ...
%         dt_tot, bt_forecast_sto_a_NC_modal_dt) ];
% end
% 
% % Reconstruction in the stochastic case
% bt_sans_coef_a_cst=bt_init;
% for l = 1:N_test
%     bt_sans_coef_a_cst = [bt_sans_coef_a_cst; ...
%         evol_forward_bt_RK4(ILC_a_cst.tot.I,ILC_a_cst.tot.L,ILC_a_cst.tot.C, ...
%         dt_tot, bt_sans_coef_a_cst) ];
% end
% bt_sans_coef_a_NC=bt_init;
% for l = 1:N_test
%     bt_sans_coef_a_NC = [bt_sans_coef_a_NC; ...
%         evol_forward_bt_RK4(ILC_a_NC.tot.I,ILC_a_NC.tot.L,ILC_a_NC.tot.C, ...
%         dt_tot, bt_sans_coef_a_NC) ];
% end


param.dt = param.dt *n_simu;
param.N_test = param.N_test /n_simu;
bt_forecast_deter = bt_forecast_deter(1:n_simu:end,:,:);
bt_forecast_MEV = bt_forecast_MEV(1:n_simu:end,:,:);
bt_forecast_EV = bt_forecast_EV(1:n_simu:end,:,:);
bt_forecast_NLMEV = bt_forecast_NLMEV(1:n_simu:end,:,:);
% bt_forecast_sto_scalar = bt_forecast_sto_scalar(1:n_simu:end,:);
% bt_forecast_sto_beta = bt_forecast_sto_beta(1:n_simu:end,:);
% bt_forecast_sto_a_cst_modal_dt = bt_forecast_sto_a_cst_modal_dt(1:n_simu:end,:);
% bt_forecast_sto_a_NC_modal_dt = bt_forecast_sto_a_NC_modal_dt(1:n_simu:end,:);
% bt_sans_coef_a_cst = bt_sans_coef_a_cst(1:n_simu:end,:);
% bt_sans_coef_a_NC = bt_sans_coef_a_NC(1:n_simu:end,:);



%% Save 3rd results, especially I, L, C and the reconstructed Chronos
% % if param.a_time_dependant
% %     dependance_on_time_of_a = '_a_time_dependant_';
% % else
% %     dependance_on_time_of_a = '_a_cst_';
% % end
% if param.decor_by_subsampl.bool
% %     if strcmp(dependance_on_time_of_a,'a_t')
% %         char_filter = [ '_on_' param.type_filter_a ];
% %     else
% %         char_filter = [];
% %     end
%         char_filter = [];
%     %         save([ param.folder_results '2ndresult_' param.type_data '_' num2str(param.nb_modes) '_modes_' ...
%     %             dependance_on_time_of_a char_filter ...
%     %             '_decor_by_subsampl_' num2str(param.decor_by_subsampl.meth) ...
%     %             '_choice_' num2str(param.decor_by_subsampl.choice_n_subsample)  ...
%     %             '_threshold_' num2str(param.decor_by_subsampl.spectrum_threshold) ...
%     %             'fct_test_' param.decor_by_subsampl.test_fct '.mat']);
%     file_save = [ param.folder_results 'EV_result_' param.type_data '_' num2str(param.nb_modes) '_modes'];
% else
%     %     save([ param.folder_results '2ndresult_' param.type_data '_' num2str(param.nb_modes) '_modes_' ...
%     %         dependance_on_time_of_a '.mat']);
%     file_save=[ param.folder_results 'EV_result_' param.type_data '_' num2str(param.nb_modes) '_modes'];
% end
file_save=[ param.folder_results 'EV_result_' param.type_data '_' num2str(param.nb_modes) '_modes'];
% if isfield(param,'N_estim')
%     file_save=[file_save '_p_estim_' num2str(param.period_estim)];
% end
if param.add_noise
    file_save=[file_save '_noise'];
end
file_save=[file_save '.mat'];
ILC_EV = ILC;
param_deter = param;
save(file_save,'param_deter',...
    'bt_forecast_deter','bt_forecast_MEV',...
    'bt_forecast_EV','bt_forecast_NLMEV',...
    'ILC_EV');
clear C_deter C_sto L_deter L_sto I_deter I_sto
% if param.big_data
disp('EV result saved');
% end

% Remove temporary files
% if isfield(param,'folder_file_U_temp' )
%     rmdir(param.folder_file_U_temp,'s');
% end


%% Plots

% param.plot.plot_deter=plot_deterministic;
% param.plot.plot_EV=plot_EV;
% param.plot.plot_tuned=plot_tuned;
% param.plot_modal_dt = plot_modal_dt;
% 
% % plot_bt_dB(param,bt_forecast_sto_scalar,bt_forecast_sto_beta,...
% %     bt_forecast_sto_a_cst_modal_dt, bt_forecast_sto_a_NC_modal_dt, bt_forecast_deter,...
% %     bt_forecast_MEV,bt_sans_coef_a_cst,bt_sans_coef_a_NC,bt_tot)
% plot_bt_dB(param,bt_forecast_EV,bt_forecast_sto_beta,...
%     bt_forecast_sto_a_cst_modal_dt, bt_forecast_sto_a_NC_modal_dt, bt_forecast_deter,...
%     bt_forecast_MEV,bt_sans_coef_a_cst,bt_sans_coef_a_NC,bt_tot)
% 
% plot_each_mode = true

% if plot_each_mode
%     plot_bt(param,bt_forecast_sto_scalar,bt_forecast_sto_beta,...
%         bt_forecast_sto_a_cst_modal_dt, bt_forecast_sto_a_NC_modal_dt, bt_forecast_deter,...
%         bt_forecast_MEV,bt_sans_coef_a_cst,bt_sans_coef_a_NC,bt_tot)
% end
