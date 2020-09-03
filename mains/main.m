function main(type_data,nb_modes,igrida,coef_correctif_estim,...
    save_all_bi,decor_by_subsampl,a_time_dependant, ...
    adv_corrected,eq_proj_div_free)
% Launch a complete simulation with a fixed set of parameters
%

global computed_PIV_variance_tensor
global compute_fake_PIV
global compute_PIV_modes

%% Reset

% init; 
% clear;
% close all;
% dbstop if error
clear param bt_forecast_sto bt_forecast_deter bt_tot
tic;

% igrida = true if the script is run on the computing grid IGRIDA
% igrida = false if the script is run on a local computer
% if nargin < 2
%     igrida = false; % do not modify this line
% end
igrida = false; % do not modify this line


%% Parameters choice

% Plots to do
plot_deterministic=true; % deterministic POD-Galerkin
plot_each_mode=false;
% plot_each_mode=true;
reconstruct_chronos = false;
% reconstruct_chronos = true;

% % Load c
% param.load_matrix_c=true;

% Number of particle for the MCMC simulation
% % param.N_particules=2;
% % param.N_particules=1000;
% param.N_particules=10;

% Rate of increase of the time step to simulate accurately the SDE
% % n_simu = 1;
% % n_simu = 100;
% n_simu = 1e3;
% % n_simu = 1e4

% Type of data used
% param.type_data = fct_type_data_choice(igrida);
param.type_data = type_data;

% Modification of the Reynolds for the Galerking projection (to test
% robustness)
if strcmp(param.type_data, 'inc3D_Re3900_blocks_truncated') ...
       || strcmp(param.type_data, 'inc3D_Re3900_blocks')
    param.modified_Re = true;
else
    param.modified_Re = false;
end

% Number of POD modes
if nargin == 0
    param.nb_modes = 2;
else
    param.nb_modes = nb_modes; clear nb_modes
end

% Compromise speed / RAM involved
% If param.big_data = true, the data are assumed to be very big.
% Thus, loops are used instead of direct computations on N-D array
param.big_data = false;

% Model for the tensor a
if ~exist('a_time_dependant','var')
    param.a_time_dependant=false;
else % the variable is already defined in a super_main
    param.a_time_dependant=a_time_dependant;
end

%% Parameters already chosen
% Do not modify the following lines

% Model for the tensor a
if param.a_time_dependant
    param.type_filter_a='b_i';
end

% Parameters used if data are saved in different files
switch param.type_data
    case {'inc3D_Re300_40dt_blocks', 'inc3D_Re300_40dt_blocks_truncated', 'inc3D_Re300_40dt_blocks_test_basis'...
            'inc3D_Re3900_blocks', 'inc3D_Re3900_blocks_truncated', 'inc3D_Re3900_blocks_test_basis'...
            'turb2D_blocks', 'turb2D_blocks_truncated', 'turb2D_blocks_test_basis',...
             'DNS300_inc3d_3D_2017_04_02_blocks', 'DNS300_inc3d_3D_2017_04_02_blocks_truncated',...
             'DNS300_inc3d_3D_2017_04_02_blocks_test_basis',...
             'test2D_blocks', 'test2D_blocks_truncated', 'test2D_blocks_test_basis',...
             'small_test_in_blocks', 'small_test_in_blocks_truncated',...
             'small_test_in_blocks_test_basis',...
             'DNS100_inc3d_2D_2018_11_16_blocks',...
             'DNS100_inc3d_2D_2018_11_16_blocks_truncated',...
             'DNS100_inc3d_2D_2018_11_16_blocks_test_basis',...
             'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks',...
             'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_truncated',...
             'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_test_basis',...
             'DNS100_OpenFOAM_2D_2020_blocks',...
             'DNS100_OpenFOAM_2D_2020_blocks_truncated',...
             'DNS100_OpenFOAM_2D_2020_blocks_test_basis',...
             'inc3D_HRLESlong_Re3900_blocks',...
             'inc3D_HRLESlong_Re3900_blocks_truncated',...
             'inc3D_HRLESlong_Re3900_blocks_test_basis'}
         
        % data_in_blocks.bool = trues if data are saved in different files
        param.data_in_blocks.bool = true;
        param.big_data=true;
%     case {
%              'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks',...
%              'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_truncated',...
%              'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_test_basis'}
%         % data_in_blocks.bool = trues if data are saved in different files
%         param.data_in_blocks.bool = true;
%         param.big_data=false;
    otherwise
        param.data_in_blocks.bool = false;
        % param.data_in_blocks.nb_blocks is the number of files used to
        % save the data
        param.data_in_blocks.nb_blocks=1;
        param.data_in_blocks.type_data2=param.type_data;
        if param.nb_modes > 16
            param.big_data=true;
        end
        param.decor_by_subsampl.bug_sampling = false ;
end

% Parameters used for the time subsampling of data
if ~exist('decor_by_subsampl','var')
    param.decor_by_subsampl.bool=true;
    if param.decor_by_subsampl.bool
        % param.decor_by_subsampl.n_subsampl_decor is the new time sampling
        % period dived by the former one.
        % param.decor_by_subsampl.n_subsampl_decor = nan if you let the
        % algorithm choose this value
        param.decor_by_subsampl.n_subsampl_decor=nan;
        %         param.decor_by_subsampl.n_subsampl_decor=10;
        % param.decor_by_subsampl.meth determines at which step of the
        % algortihm, there should be a time subsampling
        % POD_decor : at the beginning
        % bt_decor : after estimating the Chronos bt
        % a_estim_decor : after the variance tensor a
        param.decor_by_subsampl.meth='bt_decor'; % 'POD_decor' 'bt_decor' 'a_estim_decor'
        % param.decor_by_subsampl.choice_n_subsample determines the way of
        % choosing the new time sampling period
        % 'auto_shanon' means that it use a Nyquist-Shanon based criterion
        % 'tuning' means that param.decor_by_subsampl.n_subsampl_decor is
        % set manually
        % 'lms' uses the correlation time estimated through an lms filtered
        % correlation function to choose the subsampling rate.
        % 'truncated' uses the correlation time estimated through a 
        % truncated correlation function to choose the subsampling rate.
        % 'htgen' uses the correlation time estimated through an
        % heterogeneous estimator to choose the subsampling rate.
        param.decor_by_subsampl.choice_n_subsample='auto_shanon';
        % param.decor_by_subsampl.spectrum_threshold and param.decor_by_subsampl.test_fct
        % are parameters used in the new time sampling period choice.
        % spectrum_threshold is a threshold to know
        % when a Chronos spectrum is equal to zero
        param.decor_by_subsampl.spectrum_threshold=5e-4;
        %         param.decor_by_subsampl.spectrum_threshold=1e-4;
        %         param.decor_by_subsampl.spectrum_threshold=1/1000;
        % test_fct determines if the Shanon criterion is used on the derivatives
        % of Chronos ('db') or on quadratic functions of Chronos ('b')
        param.decor_by_subsampl.test_fct = 'b';%'b';'db';
    end
else % the variable is already defined in a super_main
    param.decor_by_subsampl=decor_by_subsampl;
end

% If coef_correctif_estim.learn_coef_a = true, corrective coefficients are
% estimated to modify the values of the variance tensor a, in order to
% improve the results
param.coef_correctif_estim.learn_coef_a = false;

% whether we want to add advection correction or not
param.adv_corrected = adv_corrected;

% param.eq_proj_div_free=true if the PDE is
% projected on the free divergence space
param.eq_proj_div_free = eq_proj_div_free;
% param.eq_proj_div_free = true;

% if param.save_all_bi = true, the N Chronos will be computed and saved
% with N = the number of time steps
% It can be useful for offline studies
if ~exist('save_all_bi', 'var')
    param.save_all_bi = false;
else % the variable is already defined in a super_main
    param.save_all_bi = save_all_bi;
    clear save_all_bi
end
param.save_bi_before_subsampling = true;

% If plot_bts = true the Chronos will be plotted
plot_bts = true;

% Set directories to load and save data
% if igrida
%     param.folder_data = '/temp_dd/igrida-fs1/vressegu/data/';
%     param.folder_results = '/temp_dd/igrida-fs1/vressegu/results/';
%     param.big_data =true;
%     plot_bts = false;
%     param.N_particules=1000;
%     n_simu = 100;
% else
    param.folder_results = [ pwd '/resultats/current_results/'];
    current_pwd = pwd; cd ..
    param.folder_data = [ pwd '/data/'];
    cd(current_pwd); clear current_pwd
%     param.folder_results = ['/Users/Resseguier/Documents/MATLAB/POD/all/resultats/current_results/'];
%     param.folder_data = '/Users/Resseguier/Documents/MATLAB/POD/data/';
%     param.folder_data = '/Users/vressegu/Documents/matlab/POD-NS_Stochastique/current_used/data/';
%     param.folder_results = ['/Users/vressegu/Documents/matlab/POD-NS_Stochastique/current_used/' ...
%         'all/resultats/current_results/'];
%     param.folder_results =  [ pwd '/resultats/current_results/'];
% end
param.name_file_mode=['mode_' param.type_data '_' num2str(param.nb_modes) '_modes.mat'];
param.igrida=igrida; clear igrida
param.name_file_mode=[ param.folder_data param.name_file_mode ];

%% Correction tau threshold from previous files

bool=false;
param_ref_tau=param;
param_temp = param;
param_temp.decor_by_subsampl.threshold_effect_on_tau_corrected = false;
param_temp = fct_name_1st_result_new(param_temp);
if param.decor_by_subsampl.threshold_effect_on_tau_corrected && ...
        (~param_temp.decor_by_subsampl.threshold_effect_on_tau_corrected) && ...
        (exist(param_temp.name_file_1st_result,'file')==2)
    load(param_temp.name_file_1st_result);
    param.decor_by_subsampl.threshold_effect_on_tau_corrected = ...
        param_ref_tau.decor_by_subsampl.threshold_effect_on_tau_corrected ;
    rate_dt = param.decor_by_subsampl.tau_corr ...
        / param.decor_by_subsampl.n_subsampl_decor;
    bool = true;
end
param_temp = param;
param_temp.decor_by_subsampl.threshold_effect_on_tau_corrected = true;
param_temp = fct_name_1st_result_new(param_temp);
if param_temp.decor_by_subsampl.threshold_effect_on_tau_corrected && ...
        (~param.decor_by_subsampl.threshold_effect_on_tau_corrected) && ...
        (exist(param_temp.name_file_1st_result,'file')==2)
    load(param_temp.name_file_1st_result);        
    param.decor_by_subsampl.threshold_effect_on_tau_corrected = ...
        param_ref_tau.decor_by_subsampl.threshold_effect_on_tau_corrected ;
    rate_dt = 1/( param.decor_by_subsampl.tau_corr ...
        / param.decor_by_subsampl.n_subsampl_decor );
    bool = true;
end
if bool
    I_deter=ILC.deter.I;
    L_deter=ILC.deter.L;
    C_deter=ILC.deter.C;
    I_sto=ILC.sto.I;
    L_sto=ILC.sto.L;
    C_sto=ILC.sto.C;
    
    C_sto= rate_dt * C_sto;
    L_sto= rate_dt * L_sto;
    for q=1:param.nb_modes
        I_sto(q)=-trace(diag(param.lambda)*C_sto(:,:,q));
    end
    ILC.tot.I=I_sto+I_deter;
    ILC.tot.L=L_sto+L_deter;
    ILC.tot.C=C_sto+C_deter;
    
    % Noise terms
    r_rate_dt = sqrt(rate_dt);
    pchol_cov_noises = r_rate_dt * pchol_cov_noises;
    Cov_noises = pchol_cov_noises * pchol_cov_noises';
    
    param = fct_name_1st_result_new(param);
    save(param.name_file_1st_result,'-v7.3');
    toc;tic;
    disp('1st result saved');
    return
end
clear param_temp param_ref bool;

%% POD

% Computation of the reference Chronos (bt_tot) and Topos, from the snapshots
% Choice of the time subsampling
% Conmputation of each snapshots residual velocities
% Since Topos and residual velocities are big data, they are saved in
% specific files
[param,bt_tot] = POD_and_POD_knowing_phi(param);
if param.big_data
    toc;tic;
    disp('POD done');
end

%% Quadratic variation tensor estimation
if compute_PIV_modes
    PIV_modes(param);
    toc;tic;
    disp('Fake PIV computation done');
    return;
end
if compute_fake_PIV
    Fake_PIV_data(param,true);
    Fake_PIV_data(param,false);
    toc;tic;
    disp('Fake PIV computation done');
    return;
end
global correlated_model;
if computed_PIV_variance_tensor
    quadratic_variation_estimation_PIV(param,bt_tot);
    toc;tic;
    disp('PIV variance tensor estimation done');
    return;
end
% Computation of the variance tensor a
% or computation of its parameters z_i(x)
% Since it is big data, they are saved in specific files
if ~correlated_model
    param = quadratic_variation_estimation(param,bt_tot);
    if param.big_data
        toc;tic;
        disp('Variance tensor estimation done');
    end
end

%% POD-Galerkin projection

% Force compilation
diff_l( zeros(2,2,2,1,2,2) , 1 , 1);

% Compute the coefficients of the ODE of the Chronos b(t)
% I are the constant coefficients
% L are the linear coefficients
% C are the quadratic coefficients

% Coefficents in the deterministic case
[bool_exist_ILC_deter, param, ILC_deter]= fct_exist_ILC_deter(param);
if bool_exist_ILC_deter
%     param = varargout{1};
    I_deter = ILC_deter{1};
    L_deter = ILC_deter{2};
    C_deter = ILC_deter{3};
else
    [I_deter,L_deter,C_deter,param] = param_ODE_bt_deter(param.name_file_mode, param, param.grid);
end
if param.big_data
    toc;tic;
    disp('Galerkin projection on deterministic Navier-Stokes done');
end
% Additional coefficients due to stochastic terms
global stochastic_integration
if ~ correlated_model
    if param.adv_corrected
        if strcmp(stochastic_integration, 'Ito')
            [I_sto,L_sto,C_sto] = param_ODE_bt_sto(param.name_file_mode, param, param.grid);
        elseif strcmp(stochastic_integration, 'Str')
            [I_sto,L_sto,C_sto] = param_ODE_bt_sto(param.name_file_mode, param, param.grid);
            [F1, ~] = coefficients_sto(param);
            L_sto = L_sto - F1;
        else
            error('Invalid stochastic integration path');
        end
    else
        if strcmp(stochastic_integration, 'Ito')
            [F1, ~] = coefficients_sto(param);
        elseif strcmp(stochastic_integration, 'Str')
            F1 = zeros([param.nb_modes param.nb_modes 1]);
        else
            error('Invalid stochastic integration path')
        end
        L_sto = F1;
        I_sto = zeros([param.nb_modes 1]);
        C_sto = zeros([param.nb_modes param.nb_modes param.nb_modes]);
    end
else 
    I_sto = zeros([param.nb_modes 1]);
    L_sto = zeros([param.nb_modes param.nb_modes]);   
    C_sto = zeros([param.nb_modes param.nb_modes param.nb_modes]);    
end

deter = struct('I',I_deter,'L',L_deter,'C',C_deter);
sto = struct('I',I_sto,'L',L_sto,'C',C_sto);
I_sto = I_deter + I_sto;
L_sto = L_deter + L_sto;
C_sto = C_deter + C_sto;
tot = struct('I',I_sto,'L',L_sto,'C',C_sto);
ILC=struct('deter',deter,'sto',sto,'tot',tot);
if param.big_data
    toc;tic;
    disp('Galerkin projection on stochastic Navier-Stokes done');
end

% global correlated_model;
if correlated_model
    [Cov_noises, pchol_cov_noises, eta_0, Mi_ss_0] = estimation_correlated_noises(param, bt_tot);
else
    [Cov_noises,pchol_cov_noises] = estimation_noises(param,bt_tot,ILC.tot);
end

if param.big_data
	toc;tic;
	disp('Estimation of noise correlations done');
end

%% Save first results, especially I, L, C
clear plot_modal_dt
% param = fct_name_1st_result(param);
param = fct_name_1st_result_new(param);
save(param.name_file_1st_result,'-v7.3');
clear coef_correctif_estim
% if param.igrida
    toc;tic;
    disp('1st result saved');
% end

