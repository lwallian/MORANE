function main_from_existing_ROM(nb_modes,threshold,type_data,...
    nb_period_test,...
    no_subampl_in_forecast,reconstruction,adv_corrected,modal_dt,test_fct,svd_pchol,eq_proj_div_free)
% Load simulation results, estimate modal time step by Shanon
% and compare it with modal Eddy Viscosity ROM and
% tuned version of the loaded results
%
global correlated_model
global choice_n_subsample;
global stochastic_integration;
global estim_rmv_fv;


%% Make the randomness reproducible
stream = RandStream.getGlobalStream;
reset(stream);

clear param bt_forecast_sto bt_forecast_deter bt_tot
tic

%% Parameters choice
if nargin < 9
    test_fct = 'b';
end
if nargin < 10
    svd_pchol = false;
end
if ~ strcmp(choice_n_subsample,'auto_shanon')
    modal_dt = 0;
end
% param_ref.n_simu = 2;
% N_particules=2;
param_ref.n_simu = 100;
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
% plot_EV=true; % estimated Eddy Viscosity
% plot_tuned=false; % estimated corrective coefficients

if nargin < 8
    if strcmp(type_data,'incompact3d_wake_episode3_cut_truncated')
        modal_dt=false; % different time step (inferred by Shanon criterion) for the different modes
        warning('no modal time step');
    else
        modal_dt=false; % different time step (inferred by Shanon criterion) for the different modes
        warning('no modal time step');
        %     modal_dt=true; % different time step (inferred by Shanon criterion) for the different modes
        %     warning('modal time step');
    end
end

%%

% plot_each_mode=false;

% Threshold of the Chronos spectrum, used to choice the time step
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

% On which function the Shanon criterion is used
% test_fct='b'; % 'b' is better than db

% Learning duration
% period_estim=3;
% % p_estim=13;
% % N_estim=842;
% coef_correctif_estim.learning_time='N_estim'; % 'quarter' or 'all'
% % coef_correctif_estim.learning_time='quarter'; % 'quarter' or 'all'



% % On which function the Shanon criterion is used
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
param_ref.svd_pchol=svd_pchol;
param_ref.eq_proj_div_free=eq_proj_div_free;

% if nargin > 0
%     plot_each_mode=false;
% end

modal_dt_ref = modal_dt;

%% Get data

% On which function the Shanon criterion is used
% test_fct='b'; % 'b' is better than db
a_t = '_a_cst_';
param_ref.a_time_dependant = 0; % to account for the a_t
param_ref.decor_by_subsampl.bool = true; % we'll subsample

param_ref.decor_by_subsampl.choice_n_subsample = choice_n_subsample; % for testing
param_ref.decor_by_subsampl.spectrum_threshold = threshold;
param_ref.type_data = type_data;
param_ref.nb_modes = nb_modes;
param_ref.decor_by_subsampl.meth = 'bt_decor';

param_ref.adv_corrected = adv_corrected;

param_ref.decor_by_subsampl.test_fct = test_fct;

param_ref = fct_name_1st_result_new(param_ref);
if exist(param_ref.name_file_1st_result,'file') == 2
    load(param_ref.name_file_1st_result)
else
    file_res = fct_file_save_1st_result(param_ref);
    file_res = file_res(1:end - 14); % delete the .mat at the end of the filename
    file_res=[file_res '_fullsto'];
    if ~ adv_corrected
        file_res=[file_res '_no_correct_drift'];
    end
    file_res_save = file_res;
    file_res=[ file_res '_integ_' stochastic_integration];
    if estim_rmv_fv
        file_res=[file_res '_estim_rmv_fv'];
    end
    file_res=[ file_res '.mat'];
    
    if (~(exist(file_res,'file') == 2)) ...
            && strcmp(stochastic_integration,'Ito')
        file_res = file_res_save;
        if estim_rmv_fv
            file_res=[file_res '_estim_rmv_fv'];
            param.estim_rmv_fv = true;
        end
        file_res=[file_res '.mat'];
    else
        clear file_res_save;
    end
    
    threshold
    
    % param_ref.decor_by_subsampl.test_fct = 'db';
    % param_ref.adv_corrected = adv_corrected;
    
    % file_res = fct_file_save_1st_result(param_ref);
    % file_name_struct = fct_name_1st_result(param_ref);
    % file_res = file_name_struct.name_file_1st_result;
    
    % if correlated_model
    %     file_res = file_res(1:end - 25); % delete the .mat at the end of the filename
    %     file_res=[file_res '_fullsto'];
    %     if ~ adv_corrected
    %         file_res=[file_res '_no_correct_drift'];
    %     end
    %     file_res=[file_res '_correlated'];
    %     file_res=[ file_res '_integ_' stochastic_integration];
    %     file_res=[ file_res '.mat'];
    % else
    %     file_res = file_res(1:end - 14); % delete the .mat at the end of the filename
    %     file_res=[file_res '_fullsto'];
    %     if ~ adv_corrected
    %         file_res=[file_res '_no_correct_drift'];
    %     end
    %     file_res=[ file_res '_integ_' stochastic_integration];
    %     file_res=[ file_res '.mat'];
    % end
    load(file_res)
end

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
% modal_dt

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

%% Redefined path to get access to data
param.nb_period_test=nb_period_test;
param.decor_by_subsampl.test_fct=test_fct;

svd_pchol = param_ref.svd_pchol;
folder_data = param_ref.folder_data;
folder_results = param_ref.folder_results;
% folder_data = '/Users/vressegu/Documents/matlab/POD-NS_Stochastique/current_used/data/';
% folder_results = ['/Users/vressegu/Documents/matlab/POD-NS_Stochastique/current_used/' ...
%     'all/resultats/current_results/'];

%     param.folder_results =  [ pwd '/resultats/current_results/'];
big_data=false;
% plot_bts=false;

% coef_correctif_estim=coef_correctif_estim_ref;
param.svd_pchol = svd_pchol;
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

%% Reduction of the noise matrix

if param.svd_pchol
%     figure;plot(diag(pchol_cov_noises*pchol_cov_noises'))
%     cov = pchol_cov_noises*pchol_cov_noises';
    
    % Normalizing noise covariance
    sq_lambda = sqrt(param.lambda);
%     sq_lambda = ones(size(sq_lambda));
    if correlated_model
%         sq_lambda_cat = [ sq_lambda' 1 ];
%     else
%         sq_lambda_cat = [ sq_lambda' ];
        sq_lambda_cat = [ 1 sq_lambda' 1 ];
    else
        sq_lambda_cat = [ 1 sq_lambda' ];
    end
    pchol_add_noise = pchol_cov_noises(1:param.nb_modes,:);
    pchol_cov_noises(1:param.nb_modes,:)=[];
    n_plus = length(sq_lambda_cat);
    pchol_cov_noises = reshape( pchol_cov_noises , ...
        [ (n_plus-1) param.nb_modes n_plus*param.nb_modes ] );
    
%     for q=1:(n_plus*param.nb_modes)
%         pchol_cov_noises(1:end,:,q)
%     end
    
    %     if ~correlated_model
    pchol_add_noise = permute( pchol_add_noise ,[ 3 1 2]);
    pchol_cov_noises = cat(1,pchol_add_noise,pchol_cov_noises);
    %     end
    
    
    for k=1:n_plus
        for i=1:(param.nb_modes)
%             (sq_lambda_cat(k)/sq_lambda(i))
            pchol_cov_noises(k,i,:) = (sq_lambda_cat(k)/sq_lambda(i)) * ...
                pchol_cov_noises(k,i,:);
        end
    end
    
    %     if ~correlated_model
    pchol_add_noise = pchol_cov_noises(1,:,:);
    pchol_cov_noises(1,:,:)=[];
    %     end
    
%     for q=1:(n_plus*param.nb_modes)
%         pchol_cov_noises(1:end,:,q)
%     end
    
    pchol_cov_noises = reshape( pchol_cov_noises , ...
        [ (n_plus-1)*param.nb_modes n_plus*param.nb_modes ] );
%     pchol_cov_noises = cat(1,pchol_add_noise,pchol_cov_noises);
    
    %     if ~correlated_model
    pchol_add_noise = permute( pchol_add_noise ,[ 2 3 1]);
    pchol_cov_noises = cat(1,pchol_add_noise,pchol_cov_noises);
    %     end
    
%     
%     figure;plot(diag(pchol_cov_noises*pchol_cov_noises'))
%     cov2 = pchol_cov_noises*pchol_cov_noises';
%     figure;imagesc(cov2)
%     cov-cov2
    
%     cov2 = cov;
    
    % Noise PCA
    var_pchol_cov_noises_ini = trace(pchol_cov_noises*pchol_cov_noises');
    [U_cov_noises,S_cov_noises,~] = ...
        svds(pchol_cov_noises,param.nb_modes);
    pchol_cov_noises = U_cov_noises * S_cov_noises;
    
    var_pchol_cov_noises_red = trace(pchol_cov_noises*pchol_cov_noises');
    ratio_var_red_pchol = var_pchol_cov_noises_red/var_pchol_cov_noises_ini
    
%     figure;plot(diag(pchol_cov_noises*pchol_cov_noises'))
%     cov2 = pchol_cov_noises*pchol_cov_noises';
%     figure;imagesc(cov2)
    
    % de-Normalizing noise covariance
    pchol_add_noise = pchol_cov_noises(1:param.nb_modes,:);
    pchol_cov_noises(1:param.nb_modes,:)=[];
    
    pchol_cov_noises = reshape( pchol_cov_noises , ...
        [ (n_plus-1) param.nb_modes param.nb_modes ] );
    
%     for q=1:(n_plus*param.nb_modes)
%         pchol_cov_noises(1:end,:,q)
%     end
    
    %     if ~correlated_model
    pchol_add_noise = permute( pchol_add_noise ,[ 3 1 2]);
    pchol_cov_noises = cat(1,pchol_add_noise,pchol_cov_noises);
    %     end
    
    
    for k=1:n_plus
        for i=1:(param.nb_modes)
            pchol_cov_noises(k,i,:) = (sq_lambda(i)/sq_lambda_cat(k)) * ...
                pchol_cov_noises(k,i,:);
        end
    end
    
    %     if ~correlated_model
    pchol_add_noise = pchol_cov_noises(1,:,:);
    pchol_cov_noises(1,:,:)=[];
    %     end
    
%     for q=1:(n_plus*param.nb_modes)
%         pchol_cov_noises(1:end,:,q)
%     end
    
    pchol_cov_noises = reshape( pchol_cov_noises , ...
        [ (n_plus-1)*param.nb_modes param.nb_modes ] );
%     pchol_cov_noises = cat(1,pchol_add_noise,pchol_cov_noises);
    
    %     if ~correlated_model
    pchol_add_noise = permute( pchol_add_noise ,[ 2 3 1]);
    pchol_cov_noises = cat(1,pchol_add_noise,pchol_cov_noises);
    %     end
    
    
%     figure;plot(diag(pchol_cov_noises*pchol_cov_noises'))
%     
%     figure;imagesc(cov);cax=caxis; colorbar;
%     cov2 = pchol_cov_noises*pchol_cov_noises';
%     figure;imagesc(cov2);caxis(cax); colorbar;
    
end

%% Do not temporally subsample, in order to prevent aliasing in the results
% % BETA
% if no_subampl_in_forecast & reconstruction
%     error('The reconstruction is only coded with the subsampled data');
% end
if ~ reconstruction
    %     if param.decor_by_subsampl.no_subampl_in_forecast
    %         param.dt = param.dt / param.decor_by_subsampl.n_subsampl_decor;
    %         param.N_test = param.N_test * param.decor_by_subsampl.n_subsampl_decor;
    %         param.N_tot = param.N_tot * param.decor_by_subsampl.n_subsampl_decor;
    %         param.decor_by_subsampl.n_subsampl_decor = 1;
    %     end
    
    %% Creation of the test basis
    n_subsampl_decor_ref = param.decor_by_subsampl.n_subsampl_decor;
    param.decor_by_subsampl.n_subsampl_decor = 1;
    [param,bt_tot,truncated_error2]=Chronos_test_basis(param);
    param.decor_by_subsampl.n_subsampl_decor = n_subsampl_decor_ref;
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
param.dt = param.dt/param.decor_by_subsampl.n_subsampl_decor;
% param.N_test=param.N_test*param.decor_by_subsampl.n_subsampl_decor;

%% Noise time scale
if ~correlated_model
    matcov = pchol_cov_noises * pchol_cov_noises';
    matcov = reshape(matcov,[param.nb_modes+1,param.nb_modes,param.nb_modes+1,param.nb_modes]);
    matcov(end,:,:,:)=[];
    matcov(:,:,end,:)=[];
    matcov = reshape(matcov,[param.nb_modes^2,param.nb_modes^2]);
    tau_noise = param.nb_modes / trace(matcov);
else
    matov = pchol_cov_noises * pchol_cov_noises' ;
    xi_xi= matov(1:param.nb_modes,1:param.nb_modes);
    eta_eta= matov((param.nb_modes+1):end,(param.nb_modes+1):end);
    eta_eta = reshape(eta_eta, ...
        [(param.nb_modes+1),(param.nb_modes),...
        (param.nb_modes+1),(param.nb_modes)]);
    eta_eta0 = squeeze(eta_eta((param.nb_modes+1),:,(param.nb_modes+1),:));
    eta_eta_mult = squeeze(eta_eta(1:param.nb_modes,:,1:param.nb_modes,:));
    eta_eta_mult = reshape(eta_eta_mult,...
        [param.nb_modes*param.nb_modes,...
        param.nb_modes*param.nb_modes]);
    tau_xixi = (2*mean(abs(bt_tronc(:)))/ (param.tau_ss) )^2 / ...
        ( trace(xi_xi) /param.nb_modes ) ;
    tau_eta_eta0 = (1*mean(abs(bt_tronc(:)))/ (param.tau_ss) )^2 / ...
        ( trace(eta_eta0) /param.nb_modes ) ;
    tau_eta_eta_mult = param.nb_modes / ...
        sqrt( (param.tau_ss/2) * trace(eta_eta_mult)  ) ;
    tau_noise = min([ param.tau_ss tau_xixi tau_eta_eta0 tau_eta_eta_mult ]);
end
% if tau_noise/(1e3) < param.dt
if tau_noise/(5e2) < param.dt
     warning('n_simu should be larger')
     ratio_n_simu = ceil(param.dt / (tau_noise/(5e2)));
     n_simu = ratio_n_simu * n_simu;
     param.dt = param.dt/ratio_n_simu;
     param.N_test=param.N_test*ratio_n_simu;
end

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
if strcmp(stochastic_integration, 'Ito') && ~correlated_model
    bt_MCMC=nan([param.N_test+1 param.nb_modes param.N_particules]);
    bt_MCMC(1,:,:)=repmat(bt_tronc,[1 1 param.N_particules]);
%     bt_MCMC=repmat(bt_tronc,[1 1 param.N_particules]);
%     bt_fv=bt_MCMC;
%     bt_m=zeros(1,param.nb_modes,param.N_particules);
    iii_realization = zeros(param.N_particules,1);
    for l = 1:param.N_test
%         [bt_MCMC(l+1,:,:),bt_fv(l+1,:,:),bt_m(l+1,:,:)] = ...
        bt_MCMC(l+1,:,:) = ...
            evol_forward_bt_MCMC(...
            ILC_a_cst.modal_dt.I,ILC_a_cst.modal_dt.L,ILC_a_cst.modal_dt.C, ...
            pchol_cov_noises, param.dt, bt_MCMC(l,:,:));
%             bt_fv(l,:,:),bt_m(l,:,:));
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
%                     bt_fv((l+2):param.N_test,:,:) = ...
%                         nan( param.N_test-l-1,param.nb_modes,param.N_particules);
%                     bt_m((l+2):param.N_test,:,:) = ...
%                         nan( param.N_test-l-1,param.nb_modes,param.N_particules);
                end
                break
            end
            nb_blown_up = sum(iii_realization);
            warning([ num2str(nb_blown_up) ...
                ' realizations have blown up and will be replaced.']);
            bt_MCMC_good = bt_MCMC(l+1,:, ~ iii_realization);
%             bt_fv_good = bt_fv(l+1,:, ~ iii_realization);
%             bt_m_good = bt_m(l+1,:, ~ iii_realization);
            rand_index =  randi( param.N_particules - nb_blown_up, nb_blown_up,1);
            bt_MCMC(l+1,:, iii_realization) = bt_MCMC_good(1,:, rand_index);
%             bt_fv(l+1,:, iii_realization) = bt_fv_good(1,:, rand_index);
%             bt_m(l+1,:, iii_realization) = bt_m_good(1,:, rand_index);
            clear bt_MCMC_good rand_index nb_blown_up iii_realization
        end
    end
    clear bt_tronc
    
    % warning('keeping small time step')
    param.dt = param.dt*n_simu;
    param.N_test=param.N_test/n_simu;
    bt_MCMC=bt_MCMC(1:n_simu:end,:,:);
%     bt_fv=bt_fv(1:n_simu:end,:,:);
%     bt_m=bt_m(1:n_simu:end,:,:);
    bt_forecast_sto=bt_forecast_sto(1:n_simu:end,:);
    bt_forecast_deter=bt_forecast_deter(1:n_simu:end,:);
    
    struct_bt_MCMC.tot.mean = mean(bt_MCMC,3);
    struct_bt_MCMC.tot.var = var(bt_MCMC,0,3);
    struct_bt_MCMC.tot.one_realiz = bt_MCMC(:,:,1);
    % struct_bt_MCMC.tot.one_realiz = bt_MCMC(:,:,1);
%     struct_bt_MCMC.fv.mean = mean(bt_fv,3);
%     struct_bt_MCMC.fv.var = var(bt_fv,0,3);
%     struct_bt_MCMC.fv.one_realiz = bt_fv(:,:,1);
%     struct_bt_MCMC.m.mean = mean(bt_m,3);
%     struct_bt_MCMC.m.var = var(bt_m,0,3);
%     struct_bt_MCMC.m.one_realiz = bt_m(:,:,1);
    
elseif strcmp(stochastic_integration, 'Str') && ~correlated_model
    bt_MCMC=nan([param.N_test+1 param.nb_modes param.N_particules]);
    bt_MCMC(1,:,:)=repmat(bt_tronc,[1 1 param.N_particules]);
%     bt_MCMC=repmat(bt_tronc,[1 1 param.N_particules]);
    iii_realization = zeros(param.N_particules,1);
    for l = 1:param.N_test
        [bt_MCMC(l+1,:,:)] = ...
            evol_forward_bt_SSPRK3_MCMC(...
            ILC_a_cst.modal_dt.I,ILC_a_cst.modal_dt.L,ILC_a_cst.modal_dt.C, ...
            pchol_cov_noises, param.dt, bt_MCMC(l,:,:));
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
                end
                break
            end
            nb_blown_up = sum(iii_realization);
            warning([ num2str(nb_blown_up) ...
                ' realizations have blown up and will be replaced.']);
            bt_MCMC_good = bt_MCMC(l+1,:, ~ iii_realization);
            rand_index =  randi( param.N_particules - nb_blown_up, nb_blown_up,1);
            bt_MCMC(l+1,:, iii_realization) = bt_MCMC_good(1,:, rand_index);
            clear bt_MCMC_good rand_index nb_blown_up iii_realization
        end
    end
    clear bt_tronc
    
    % warning('keeping small time step')
    param.dt = param.dt*n_simu;
    param.N_test=param.N_test/n_simu;
    bt_MCMC=bt_MCMC(1:n_simu:end,:,:);
    bt_forecast_sto=bt_forecast_sto(1:n_simu:end,:);
    bt_forecast_deter=bt_forecast_deter(1:n_simu:end,:);
    
    struct_bt_MCMC.tot.mean = mean(bt_MCMC,3);
    struct_bt_MCMC.tot.var = var(bt_MCMC,0,3);
    struct_bt_MCMC.tot.one_realiz = bt_MCMC(:,:,1);
    % struct_bt_MCMC.tot.one_realiz = bt_MCMC(:,:,1);
elseif correlated_model
    bt_MCMC = nan([param.N_test/n_simu+1 param.nb_modes param.N_particules]);
    bt_MCMC(1,:,:)=repmat(bt_tronc,[1 1 param.N_particules]);
    bt_MCMC_previous=repmat(bt_tronc,[1 1 param.N_particules]);
    struct_bt_MCMC = fct_init_struct_bt_MCMC(param,bt_MCMC_previous);
    %     struct_bt_MCMC.tot.mean=nan([param.N_test+1 param.nb_modes ]);
%     struct_bt_MCMC.tot.var=nan([param.N_test+1 param.nb_modes ]);
%     struct_bt_MCMC.tot.one_realiz=nan([param.N_test+1 param.nb_modes ]);
%     struct_bt_MCMC.qtl=nan([param.N_test+1 param.nb_modes ]);
%     struct_bt_MCMC.diff=nan([param.N_test+1 param.nb_modes ]);
%     bt_MCMC=nan([param.N_test+1 param.nb_modes param.N_particules]);
%     bt_MCMC(1,:,:)=repmat(bt_tronc,[1 1 param.N_particules]);
%     bt_fv = bt_MCMC;
%     bt_m = zeros(1, param.nb_modes, param.N_particules);
    
    % Initialization of model's stochastic variables
    eta_previous = repmat(permute(eta_0, [3, 1, 2, 4]), ...
        [1 1 1 param.N_particules]);
    struct_bt_MCMC.eta = fct_init_struct_bt_MCMC(param,eta_previous);
    Mi_ss_previous = repmat(permute(Mi_ss_0, [3, 1, 2, 4]), ...
        [1 1 param.N_particules]);
    struct_bt_MCMC.Mi_ss = fct_init_struct_bt_MCMC(param,Mi_ss_previous);
    spiral_previous = randn(1, 1, param.N_particules);
    struct_bt_MCMC.spiral = fct_init_struct_bt_MCMC(param,spiral_previous);
%     eta_0 = permute(eta_0, [3, 1, 2, 4]);
%     eta = repmat(eta_0, [1, 1, 1, param.N_particules]);
%     spiral = randn(1, 1, param.N_particules);
%     Mi_ss_0 = permute(Mi_ss_0, [3, 1, 2, 4]);
%     Mi_ss = repmat(Mi_ss_0, [1, 1, 1, param.N_particules]);
    
    for l = 1 : param.N_test
%         [bt_MCMC(l + 1, :, :), bt_fv(l + 1, :, :), bt_m(l + 1, :, :), ...
%             eta(l + 1, :, :, :), Mi_ss(l + 1, :, :), spiral(l + 1, :, :)] = ...
%             evol_forward_correlated_centered(ILC_a_cst.modal_dt.I,ILC_a_cst.modal_dt.L,ILC_a_cst.modal_dt.C, ...
%             pchol_cov_noises, param.tau_ss, param.dt, bt_MCMC(l, :, :), ...
%             eta(l, :, :, :), spiral(l, :, :), Mi_ss(l, :, :), bt_fv(l, :, :), bt_m(l, :, :));
        [bt_MCMC_new, eta_new, Mi_ss_new, spiral_new] = ...
            evol_forward_correlated_centered(ILC_a_cst.modal_dt.I,ILC_a_cst.modal_dt.L,ILC_a_cst.modal_dt.C, ...
            pchol_cov_noises, param.tau_ss, param.dt, bt_MCMC_previous, ...
            eta_previous, spiral_previous, Mi_ss_previous);
        struct_bt_MCMC = fct_fill_struct_bt_MCMC(param, l+1,...
                            struct_bt_MCMC, bt_MCMC_new);
        struct_bt_MCMC.eta = fct_fill_struct_bt_MCMC(param, l+1,...
                            struct_bt_MCMC.eta, eta_new);
        struct_bt_MCMC.Mi_ss = fct_fill_struct_bt_MCMC(param, l+1,...
                            struct_bt_MCMC.Mi_ss, Mi_ss_new);
        struct_bt_MCMC.spiral = fct_fill_struct_bt_MCMC(param, l+1,...
                            struct_bt_MCMC.spiral, spiral_new);
        bt_MCMC_previous = bt_MCMC_new;
        eta_previous = eta_new;
        Mi_ss_previous = Mi_ss_new;
        spiral_previous = spiral_new;
        if mod(l+1-1,n_simu) == 0
            l_subsampl = (l/n_simu)+1;
            bt_MCMC(l_subsampl,:,:) = bt_MCMC_new;
        end
    end
    clear bt_tronc
    
    param.dt = param.dt * n_simu;
    param.N_test = param.N_test / n_simu;
    struct_bt_MCMC = fct_subsampl_struct_bt_MCMC(param,n_simu,...
                                struct_bt_MCMC);
    struct_bt_MCMC.eta = fct_subsampl_struct_bt_MCMC(param,n_simu,...
                                struct_bt_MCMC.eta);
    struct_bt_MCMC.Mi_ss = fct_subsampl_struct_bt_MCMC(param,n_simu,...
                                struct_bt_MCMC.Mi_ss);
    struct_bt_MCMC.spiral = fct_subsampl_struct_bt_MCMC(param,n_simu,...
                                struct_bt_MCMC.spiral);
%     bt_MCMC = bt_MCMC(1 : n_simu : end, :, :);
% %     bt_fv = bt_fv(1 : n_simu : end, :, :);
% %     bt_m = bt_m(1 : n_simu : end, :, :);
%     eta = eta(1 : n_simu : end, :, :, :);
%     spiral = spiral(1 : n_simu : end, :, :);
%     Mi_ss = Mi_ss(1: n_simu : end, :, :);
    bt_forecast_sto = bt_forecast_sto(1 : n_simu : end, :);
    bt_forecast_deter = bt_forecast_deter(1 : n_simu : end, :);
    
%     struct_bt_MCMC.tot.mean = mean(bt_MCMC, 3);
%     struct_bt_MCMC.tot.var = var(bt_MCMC, 0, 3);
%     struct_bt_MCMC.tot.one_realiz = bt_MCMC(:, :, 1);
% %     struct_bt_MCMC.fv.mean = mean(bt_fv, 3);
% %     struct_bt_MCMC.fv.var = var(bt_fv, 0, 3);
% %     struct_bt_MCMC.fv.one_realiz = bt_fv(:, :, 1);
% %     struct_bt_MCMC.m.mean = mean(bt_m, 3);
% %     struct_bt_MCMC.m.var = var(bt_m, 0, 3);
% %     struct_bt_MCMC.m.one_realiz = bt_m(:, :, 1);
else
    error('Invalid stochastic integration path')
end

% BETA : confidence interval
% struct_bt_MCMC.qtl = fx_quantile(bt_MCMC, 0.025, 3);
% struct_bt_MCMC.diff = fx_quantile(bt_MCMC, 0.975, 3) - struct_bt_MCMC.qtl;
% end BETA
if param.igrida
    toc;tic
    disp('Reconstruction/Forecast of Chronos done');
end

%% Save 2nd results, especially I, L, C and the reconstructed Chronos

param = fct_name_2nd_result_new(param,modal_dt,reconstruction);
% param = fct_name_2nd_result(param,modal_dt,reconstruction);
save(param.name_file_2nd_result,'-v7.3');
% save(param.name_file_1st_result,'-v7.3');
clear C_deter C_sto L_deter L_sto I_deter I_sto
if param.igrida
    toc;tic;
    disp('2nd result saved');
end

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

