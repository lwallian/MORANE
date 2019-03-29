function main_from_existing_ROM(nb_modes,threshold,type_data,...
    nb_period_test,...
    no_subampl_in_forecast,reconstruction,adv_corrected,modal_dt)
% Load simulation results, estimate modal time step by Shanon
% and compare it with modal Eddy Viscosity ROM and
% tuned version of the loaded results
%



%% Make the randomness reproductible
stream = RandStream.getGlobalStream;
reset(stream);

clear param bt_forecast_sto bt_forecast_deter bt_tot
tic

%% Parameters choice
% param_ref.n_simu = 2;
% N_particules=2;
param_ref.n_simu = 1e2;
N_particules=100;
param_ref.N_particules=N_particules;

%% Default parameters 
% Number of POD modes
if nargin == 0
    nb_modes = 2;
end


% Type of data
if nargin < 3
    % These 3D data give good results
    % They are saved in only one file
    % (~ 250 time step)
    % type_data = 'LES_3D_tot_sub_sample_blurred';
    % type_data = 'incompact3d_wake_episode3_cut';
    type_data = 'inc3D_Re3900_blocks_truncated';
    % type_data = 'inc3D_Re3900_blocks';
    %     type_data = 'incompact3D_noisy2D_40dt_subsampl';
    
    % These 3D data are bigger, since the spatial grid is thinner
    % and the number of time step is bigger
    % (~ 2000 time step)
    % They are saved in different files
    % The internship should try to use these data
    %     type_data = 'inc3D_Re3900_blocks';
end

% Plots to do
% plot_deterministic=true; % deterministic POD-Galerkin
% plot_EV=true; % estimated Eddy Visocvity
% plot_tuned=false; % estimated corrective coefficients

if nargin < 8
    if strcmp(type_data,'incompact3d_wake_episode3_cut_truncated')
        modal_dt=false; % different time step (infered by Shanon criterion) for the different modes
        warning('no modal time step');
    else
        modal_dt=false; % different time step (infered by Shanon criterion) for the different modes
        warning('no modal time step');
        %     modal_dt=true; % different time step (infered by Shanon criterion) for the different modes
        %     warning('modal time step');
    end
end

%%

% plot_each_mode=false;

% Threshold of the Chonos spectrum, used to choice the time step
% threshold=0.001; % 0.001 or 0.01 for LES 3900 (or inc3D 3900)
% threshold=0.005; % for LES 3900
% threshold=0.0005; % for inc3D episode 3
% threshold=0.00014; % for inc3D 3900
% threshold=0.000135; % for inc3D 3900

% % Number of particle for the MCMC simulation
% %param_ref.N_particules=2;
% % param_ref.N_particules= min(1000, 100*nb_modes);
% % % param_ref.N_particules=100*nb_modes;
% % % param_ref.N_particules=1000;
% % % % param_ref.N_particules=2;
% % % % param.N_particules=2;
% % % % % param.N_particules=1000;
% N_particules=100
% % N_particules=4 % VALUE BY DEFAULT
% % N_particules=2
% % warning('only 4 particles');
% param_ref.N_particules=N_particules;

% % Rate of increase of the time step to simulate accurately the SDE
% % if strcmp( type_data,'DNS100_inc3d_2D_2018_11_16_blocks_truncated')
% %     param_ref.n_simu = 1e1;    
% % else
%     % param_ref.n_simu = 1e7;
%     % param_ref.n_simu = 1e4;
%     param_ref.n_simu = 1e2;
%     % param_ref.n_simu = 1e3;
%     % param_ref.n_simu = 1;
%     % n_simu = 1;
%     % % n_simu = 100;
% % end

% On which function the Shanon ctriterion is used
% test_fct='b'; % 'b' is better than db

% Learning duration
% period_estim=3;
% % p_estim=13;
% % N_estim=842;
% coef_correctif_estim.learning_time='N_estim'; % 'quarter' or 'all'
% % coef_correctif_estim.learning_time='quarter'; % 'quarter' or 'all'



% % On which function the Shanon ctriterion is used
% decor_by_subsampl.test_fct = 'b';

%% Parameters already chosen
% Do not modify the following lines

% coef_correctif_estim.learn_coef_a=true; % true false
% coef_correctif_estim.type_estim='vector_b'; % 'scalar' 'vector_z' 'vector_b' 'matrix'
% coef_correctif_estim.beta_min=-inf; % -inf 0 1
% coef_correctif_estim.nb_modes_used=eval('nb_modes'); % 2 eval('nb_modes') for learning the coefficient

folder_results = [ pwd '/resultats/current_results/'];
current_pwd = pwd; cd ..
folder_data = [ pwd '/data/' ];
cd(current_pwd);
% folder_results = ['/Users/Resseguier/Documents/MATLAB/POD/all/resultats/current_results/'];
% folder_data = '/Users/Resseguier/Documents/MATLAB/POD/data/';
% folder_results = ['/Users/vressegu/Documents/matlab/POD-NS_Stochastique/current_used/' ...
%     'all/resultats/current_results/'];
%     param.folder_results =  [ pwd '/resultats/current_results/'];
param_ref.folder_results=folder_results;
param_ref.folder_data =folder_data ;

% if nargin > 0
%     plot_each_mode=false;
% end

modal_dt_ref = modal_dt;

%% Get data

% On which function the Shanon ctriterion is used
test_fct='b'; % 'b' is better than db
a_t='_a_cst_';

% file_res=[ folder_results '2ndresult_' type_data '_' num2str(nb_modes) '_modes_' ...
file_res=[ folder_results '1stresult_' type_data '_' num2str(nb_modes) '_modes_' ...
    a_t '_decor_by_subsampl_bt_decor_choice_auto_shanon_threshold_' ...
    num2str(threshold) ...
    'fct_test_' test_fct ];
if exist('period_estim','var')
    file_res=[file_res '_p_estim_' num2str(period_estim)];
end

file_res=[file_res '_fullsto'];

if ~ adv_corrected
    file_res=[file_res '_no_correct_drift'];    
end

file_res=[ file_res '.mat'];
load(file_res)


param.decor_by_subsampl.no_subampl_in_forecast = no_subampl_in_forecast;

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

%% Parameters of the ODE of the b(t)
modal_dt = modal_dt_ref;
modal_dt

tot = struct('I',I_sto,'L',L_sto,'C',C_sto);

I_sto = I_sto - I_deter;
L_sto = L_sto - L_deter;
C_sto = C_sto - C_deter;

% L_deter = zeros(size(L_deter));

deter = struct('I',I_deter,'L',L_deter,'C',C_deter);
sto = struct('I',I_sto,'L',L_sto,'C',C_sto);
ILC=struct('deter',deter,'sto',sto,'tot',tot);
% ILC=struct('deter',deter,'sto',sto);

ILC_a_cst=ILC;
% bt_sans_coef_a_cst = bt_forecast_sto;

%% Redefined path to get acces to data
param.nb_period_test=nb_period_test;
param.decor_by_subsampl.test_fct=test_fct;

folder_data = param_ref.folder_data;
folder_results = param_ref.folder_results;
% folder_data = '/Users/vressegu/Documents/matlab/POD-NS_Stochastique/current_used/data/';
% folder_results = ['/Users/vressegu/Documents/matlab/POD-NS_Stochastique/current_used/' ...
%     'all/resultats/current_results/'];

%     param.folder_results =  [ pwd '/resultats/current_results/'];
big_data=false;
% plot_bts=false;

% coef_correctif_estim=coef_correctif_estim_ref;

param.folder_data = folder_data;
param.folder_results = folder_results;
param.big_data=big_data;
param.plot_bts=plot_bts;
% param.coef_correctif_estim=coef_correctif_estim;

%% Choice of modal time step

if modal_dt >0
% if modal_dt
    [rate_dt, ILC_a_cst,pchol_cov_noises] = fct_cut_frequency_2_full_sto( ...
        bt_tot,ILC_a_cst,param,pchol_cov_noises, modal_dt);
    % [rate_dt, ILC_a_cst] = fct_cut_frequency_2(bt_tot,ILC_a_cst,param);
    % [rate_dt, ILC_a_NC] = fct_cut_frequency_2(bt_tot,ILC_a_NC,param);
else
    ILC_a_cst.modal_dt = ILC_a_cst.tot;
end

%% Sensibility
% coef_mod = 1;
% if coef_mod ~= 1
%     warning('Coefficients modified to study sentibility');
% end
% param.coef_sensitivity = coef_mod;
% I_deter=ILC_a_cst.deter.I;
% L_deter=ILC_a_cst.deter.L;
% C_deter=ILC_a_cst.deter.C;
%
% I_sto= coef_mod^2 * ILC_a_cst.sto.I;
% L_sto= coef_mod^2 * ILC_a_cst.sto.L;
% C_sto= coef_mod^2 * ILC_a_cst.sto.C;
% pchol_cov_noises = coef_mod * pchol_cov_noises;
% 
% ILC_a_cst.modal_dt.I=I_sto+I_deter;
% ILC_a_cst.modal_dt.L=L_sto+L_deter;
% ILC_a_cst.modal_dt.C=C_sto+C_deter;

%% Choice the learning duration
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
% param.N_learn_coef_a=size(bt_tot,1);

%% Learning corrective coefficients
% 
% [coef_beta, ILC_beta] = estim_vector_mat_beta(bt_tot,ILC,param);
% param.coef_correctif_estim.type_estim='scalar';
% [coef_scalar, ILC_scalar] = estim_vector_mat_beta(bt_tot,ILC,param);
% [MEV, ILC] = estim_modal_eddy_viscosity(bt_tot,ILC,param);
% [NLMEV, ILC] = estim_modal_non_lin_eddy_viscosity(bt_tot,ILC,param);


%% Duration of the test 

% % warning('nb periods changed by hands')
% % param.type_data = [param.type_data '++'];
% % nb_period_test = 5*nb_period_test;
% 
% if strcmp(param.type_data, 'inc3D_Re3900_blocks')
%     param.N_test = ceil(5*5/param.dt);
%     warning('simulation on only 5 periods')
% %     param.N_test = ceil(10*5/param.dt);
% %     warning('simulation on only 10 periods')
% end
% if nargin > 3
%     param.nb_period_test=nb_period_test;
%     if strcmp(param.type_data,'incompact3d_wake_episode3_cut_truncated')
%         T_period = 4.1;
%         param.N_test = ceil( T_period * nb_period_test/param.dt);
%         warning(['simulation on only ' num2str(nb_period_test) ' periods']);
% %     else
% %         T_period = 5;
%     end
% %     param.N_test = ceil( T_period * nb_period_test/param.dt);
% %     warning(['simulation on only ' num2str(nb_period_test) ' periods']);    
% end

%% Do not temporally subsample, in order to prevent aliasing in the results
% % BETA
% if no_subampl_in_forecast & reconstruction
%     error('The recosntruction is only coded with the subsampled data');
% end
if ~ reconstruction
%     if param.decor_by_subsampl.no_subampl_in_forecast
%         param.dt = param.dt / param.decor_by_subsampl.n_subsampl_decor;
%         param.N_test = param.N_test * param.decor_by_subsampl.n_subsampl_decor;
%         param.N_tot = param.N_tot * param.decor_by_subsampl.n_subsampl_decor;
%         param.decor_by_subsampl.n_subsampl_decor = 1;
%     end
    
    %% Creation of the test basis
    [param,bt_tot,truncated_error2]=Chronos_test_basis(param);
    if param.big_data
        toc;tic;
        disp('Creation of the test basis done');
    end
end

%% Time integration of the reconstructed Chronos b(t)

param.folder_results=param_ref.folder_results;

param.N_particules=param_ref.N_particules;
n_simu=param_ref.n_simu;

param.N_tot = size(bt_tot,1);
param.N_test = param.N_tot-1;
bt_tot=bt_tot(1:(param.N_test+1),:); % reference Chronos
bt_tronc=bt_tot(1,:); % Initial condition

param.dt = param.dt/n_simu;
param.N_test=param.N_test*n_simu;
% % BETA only for test2D
% if strcmp(param.type_data , 'test2D_blocks_truncated')
%     param.N_test = 6e4-1;
% end
% %end BETA
% Reconstruction in the deterministic case
bt_forecast_deter=bt_tronc;
for l = 1:param.N_test
    bt_forecast_deter= [bt_forecast_deter; ...
        evol_forward_bt_RK4( ...
        ILC_a_cst.deter.I,ILC_a_cst.deter.L,ILC_a_cst.deter.C, ...
        param.dt, bt_forecast_deter)];
end

% Reconstruction in the stochastic case
bt_forecast_sto=bt_tronc;
for l = 1:param.N_test
    bt_forecast_sto = [bt_forecast_sto; ...
        evol_forward_bt_RK4(...
        ILC_a_cst.modal_dt.I,ILC_a_cst.modal_dt.L,ILC_a_cst.modal_dt.C, ...
        param.dt, bt_forecast_sto) ];
%         ILC_a_cst.tot.I,ILC_a_cst.tot.L,ILC_a_cst.tot.C, ...
%         param.dt, bt_forecast_sto) ];
end

% param.dt = param.dt/n_simu;
% param.N_test=param.N_test*n_simu;

% Reconstruction in the stochastic case
bt_MCMC=repmat(bt_tronc,[1 1 param.N_particules]);
bt_fv=bt_MCMC;
bt_m=zeros(1,param.nb_modes,param.N_particules);
iii_realization = zeros(param.N_particules,1);
for l = 1:param.N_test
    [bt_MCMC(l+1,:,:),bt_fv(l+1,:,:),bt_m(l+1,:,:)] = ...
        evol_forward_bt_MCMC(...
        ILC_a_cst.modal_dt.I,ILC_a_cst.modal_dt.L,ILC_a_cst.modal_dt.C, ...
        pchol_cov_noises, param.dt, bt_MCMC(l,:,:), ...
        bt_fv(l,:,:),bt_m(l,:,:));
%         ILC_a_cst.tot.I,ILC_a_cst.tot.L,ILC_a_cst.tot.C, ...
%         pchol_cov_noises, param.dt, bt_MCMC(l,:,:), ...
%         bt_fv(l,:,:),bt_m(l,:,:));

    iii_realization =  permute( any( ...
        isnan( bt_MCMC(l+1,:,:) ) | isinf( bt_MCMC(l+1,:,:) ) ...
        , 2) ,[3 1 2]); % N_particules
    if any(iii_realization)
        if all(iii_realization)
            warning('all realization of the simulation have blown up.')
            if l < param.N_test
                bt_MCMC((l+2):param.N_test,:,:) = ...
                    nan( param.N_test-l-1,param.nb_modes,param.N_particules);
                bt_fv((l+2):param.N_test,:,:) = ...
                    nan( param.N_test-l-1,param.nb_modes,param.N_particules);
                bt_m((l+2):param.N_test,:,:) = ...
                    nan( param.N_test-l-1,param.nb_modes,param.N_particules);
            end
            break
        end
        nb_blown_up = sum(iii_realization);
        warning([ num2str(nb_blown_up) ...
            ' realizations have blown up and will be replaced.']);
        bt_MCMC_good = bt_MCMC(l+1,:, ~ iii_realization);
        bt_fv_good = bt_fv(l+1,:, ~ iii_realization);
        bt_m_good = bt_m(l+1,:, ~ iii_realization);
        rand_index =  randi( param.N_particules - nb_blown_up, nb_blown_up,1);
        bt_MCMC(l+1,:, iii_realization) = bt_MCMC_good(1,:, rand_index);   
        bt_fv(l+1,:, iii_realization) = bt_fv_good(1,:, rand_index);   
        bt_m(l+1,:, iii_realization) = bt_m_good(1,:, rand_index);   
        clear bt_MCMC_good rand_index nb_blown_up iii_realization
    end
end
clear bt_tronc

% warning('keeping small time step')
param.dt = param.dt*n_simu;
param.N_test=param.N_test/n_simu;
bt_MCMC=bt_MCMC(1:n_simu:end,:,:);
bt_fv=bt_fv(1:n_simu:end,:,:);
bt_m=bt_m(1:n_simu:end,:,:);
bt_forecast_sto=bt_forecast_sto(1:n_simu:end,:);
bt_forecast_deter=bt_forecast_deter(1:n_simu:end,:);

struct_bt_MCMC.tot.mean = mean(bt_MCMC,3);
struct_bt_MCMC.tot.var = var(bt_MCMC,0,3);
struct_bt_MCMC.tot.one_realiz = bt_MCMC(:,:,1);
% struct_bt_MCMC.tot.one_realiz = bt_MCMC(:,:,1);
struct_bt_MCMC.fv.mean = mean(bt_fv,3);
struct_bt_MCMC.fv.var = var(bt_fv,0,3);
struct_bt_MCMC.fv.one_realiz = bt_fv(:,:,1);
struct_bt_MCMC.m.mean = mean(bt_m,3);
struct_bt_MCMC.m.var = var(bt_m,0,3);
struct_bt_MCMC.m.one_realiz = bt_m(:,:,1);

% BETA : confidence interval
% struct_bt_MCMC.qtl = quantile(bt_MCMC, 0.025, 3);
% struct_bt_MCMC.diff = quantile(bt_MCMC, 0.975, 3) - struct_bt_MCMC.qtl;
% end BETA
if param.igrida
    toc;tic
    disp('Reconstruction/Forecast of Chronos done');
end

%% Save 2nd results, especially I, L, C and the reconstructed Chronos

param = fct_name_2nd_result(param,modal_dt,reconstruction);
save(param.name_file_2nd_result,'-v7.3');
% save(param.name_file_1st_result,'-v7.3');
clear C_deter C_sto L_deter L_sto I_deter I_sto
if param.igrida
    toc;tic;
    disp('2nd result saved');
end
% 
% if param.decor_by_subsampl.bool
%     if strcmp(dependance_on_time_of_a,'a_t')
%         char_filter = [ '_on_' param.type_filter_a ];
%     else
%         char_filter = [];
%     end
%     file_save=[ param.folder_results '2ndresult_' param.type_data '_' num2str(param.nb_modes) '_modes_' ...
%         dependance_on_time_of_a char_filter ...
%         '_decor_by_subsampl_' num2str(param.decor_by_subsampl.meth) ...
%         '_choice_' num2str(param.decor_by_subsampl.choice_n_subsample)  ...
%         '_threshold_' num2str(param.decor_by_subsampl.spectrum_threshold) ...
%         'fct_test_' param.decor_by_subsampl.test_fct ];
% else
%     file_save=[ param.folder_results '2ndresult_' param.type_data '_' num2str(param.nb_modes) '_modes_' ...
%         dependance_on_time_of_a ];
% end
% file_save=[file_save '_fullsto'];
% if modal_dt
%     file_save=[file_save '_modal_dt'];
% end
% if ~ adv_corrected
%     file_save=[file_save '_no_correct_drift'];    
% end
% if no_subampl_in_forecast
%     file_save=[file_save '_no_subampl_in_forecast'];
% end
% if reconstruction
%     param.reconstruction=true;
%     file_save=[file_save '_reconstruction'];
% else
%     param.reconstruction=false;
% end
% file_save=[file_save '.mat'];
% save(file_save,'-v7.3');
% % save(file_save);
% clear C_deter C_sto L_deter L_sto I_deter I_sto
% % if param.big_data
%     toc;tic
%     disp('2nd result saved');
% % end

%% Plots of the reconstructed Chronos
% for i=1:size(bt_MCMC,2)
%         figure;plot(bt_MCMC(:,i,1))
%         hold on;
%         plot(bt_tot(:,i),'r')
%         hold off;
% end
% % for i=1:size(bt_MCMC,2)
% %     for j=1:size(bt_MCMC,3)
% %         figure;plot(bt_MCMC(:,i,j))
% %     end
% % end

% plot_bts = false;

% if plot_bts
%     param.folder_data =param_ref.folder_data ;
%     
%     param.plot.plot_deter=plot_deterministic;
%     param.plot.plot_EV=plot_EV;
%     param.plot.plot_tuned=plot_tuned;
%     param.plot_modal_dt = false;
% %     param.plot_modal_dt = plot_modal_dt;
%     
% %     plot_bt_dB_MCMC(param,bt_tot,bt_tot,...
% %             bt_tot, bt_tot, bt_forecast_deter,...
% %             bt_tot,bt_forecast_sto,bt_forecast_sto,bt_tot,struct_bt_MCMC)
%     
%     zzz = nan(size(bt_tot));
%     param.plot.plot_EV = false;
%     
%     param.test_basis = true;
%     %     param.folder_results = [param.folder_results '_test_basis'];
%     plot_each_mode = true;
%         
%     if plot_each_mode
%         plot_bt_MCMC(param,zzz,zzz,...
%             zzz, zzz, bt_forecast_deter,...
%             zzz,bt_forecast_sto,zzz,bt_tot,struct_bt_MCMC)
%         figure;
%     end
% 
% %     plot_bt_dB_MCMC(param,zzz,zzz,...
%     plot_bt_dB_MCMC_varying_error(param,zzz,zzz,...
%             zzz, zzz, bt_forecast_deter,...
%             zzz,bt_forecast_sto,zzz,bt_tot,struct_bt_MCMC,bt_MCMC)
%     
% 
% % if plot_bts
% %     plot_bt5(param,bt_forecast_sto,bt_forecast_deter,bt_tot)
% % end
%     toc;tic
%     disp('plot done');
% end
    
