function main_finite_variation(nb_modes,igrida,coef_correctif_estim,...
    save_all_bi,decor_by_subsampl,type_data,a_time_dependant,N_estim)
% Launch a complete simulation with a fixed set of parameters
%

%% Reset

% init;
% clear;
% close all;
% dbstop if error
clear param bt_forecast_sto bt_forecast_deter bt_tot
tic;

% igrida = true if the script is run on the computing grid IGRIDA
% igrida = false if the script is run on a local computer
if nargin < 2
    igrida = false; % do not modify this line
end

%% Parameters choice

% Plots to do
plot_deterministic=true; % deterministic POD-Galerkin
plot_each_mode=true;

% Type of data used
% param.type_data = fct_type_data_choice(igrida);
param.type_data = type_data; clear type_data

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

% Number of snapshots used for the estimation of a
% N_estim=10;
if exist('N_estim','var')
    param.N_estim=N_estim;
end

% Model for the tensor a
if param.a_time_dependant
    param.type_filter_a='b_i';
end


% Parameters used if data are saved in different files
switch param.type_data
    %     case {'inc3D_Re300_40dt_blocks','inc3D_Re3900_blocks'}
    case {'inc3D_Re3900_blocks'}
        % data_in_blocks.bool = trues if data are saved in different files
        param.data_in_blocks.bool = true;
        param.big_data=true;
    case {'inc3D_Re300_40dt_blocks'}
        % data_in_blocks.bool = trues if data are saved in different files
        param.data_in_blocks.bool = true;
                param.big_data=true;
        param.big_data=false;
        
        %     case {'incompact3D_noisy2D_40dt_subsampl'}
        %         % data_in_blocks.bool = trues if data are saved in different files
        %         param.data_in_blocks.bool = false;
        %         param.big_data=true;
        % %         param.big_data=false;
    case {'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks'}
        % data_in_blocks.bool = trues if data are saved in different files
        param.data_in_blocks.bool = true;
        param.big_data=false;
    case {'inc3D_HRLESlong_Re3900_blocks'}
        % data_in_blocks.bool = trues if data are saved in different files
        param.data_in_blocks.bool = true;
        param.big_data=false;
        warning('big_data = false');
    case {'DNS300_inc3d_3D_2017_04_02_blocks'}
        % data_in_blocks.bool = trues if data are saved in different files
        param.data_in_blocks.bool = true;
        param.big_data=false;
        warning('big_data = false');
    otherwise
        if strcmp(param.type_data(end-5:end),'blocks')
            % data_in_blocks.bool = trues if data are saved in different files
            param.data_in_blocks.bool = true;
            param.big_data=false;
        else
            param.data_in_blocks.bool = false;
            % param.data_in_blocks.nb_blocks is the number of files used to
            % save the data
            param.data_in_blocks.nb_blocks=1;
            param.data_in_blocks.type_data2=param.type_data;
        end
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
        param.decor_by_subsampl.choice_n_subsample='auto_shanon';
        % param.decor_by_subsampl.spectrum_threshold and param.decor_by_subsampl.test_fct
        % are parameters used in the new time sampling period choice.
        % spectrum_threshold is a threshold to know
        % when a Chronos spectrum is equal to zero
        param.decor_by_subsampl.spectrum_threshold=1e-4;
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
param.coef_correctif_estim.learn_coef_a=false;

% param.eq_proj_div_free=true if the PDE is
% projected on the free divergence space
param.eq_proj_div_free=true;

% if param.save_all_bi = true, the N Chronos will be computed and saved
% with N = the number of time steps
% It can be useful for offline studies
if ~exist('save_all_bi','var')
    param.save_all_bi=false;
else % the variable is already defined in a super_main
    param.save_all_bi=save_all_bi;
    clear save_all_bi
end

% If plot_bts = true the Chronos will be plotted
plot_bts = true;

% Set directories to load and save data
if igrida
    param.folder_data = '/temp_dd/igrida-fs1/vressegu/data/';
    param.folder_results = '/temp_dd/igrida-fs1/vressegu/results/';
    param.big_data =true;
    plot_bts = false;
    plot_each_mode=false;
else
    %     param.folder_data = '/Users/vressegu/Documents/matlab/POD-NS_Stochastique/current_used/data/';
    %     param.folder_results = ['/Users/vressegu/Documents/matlab/POD-NS_Stochastique/current_used/' ...
    %         'all/resultats/current_results/'];
    % %     param.folder_results =  [ pwd '/resultats/current_results/'];
    cd ..
    param.folder_data =  [ pwd '/data/'];
    cd 'all'
    param.folder_results =  [ pwd '/resultats/current_results/'];
end
param.name_file_mode=['mode_' param.type_data '_' num2str(param.nb_modes) '_modes.mat'];
param.igrida=igrida; clear igrida
param.name_file_mode=[ param.folder_data param.name_file_mode ];


%% POD
% Computation of the reference Chronos (bt_tot) and Topos, from the snapshots
% Choice of the time subsampling
% Conmputation of each snapshots residual velocities
% Since Topos and residual velocities are big data, they are saved in
% specific files
[param,bt_tot]=POD_and_POD_knowing_phi(param);
% if param.big_data
toc;tic;
disp('POD done');
% end

%% Quadratique variation tensor estimation

% Computation of the variance tensor a
% or computation of its parameters z_i(x)
% Since it is big data, they are saved in specific files
param = quadratic_variation_estimation(param,bt_tot);
% if param.big_data
toc;tic;
disp('Variance tensor estimation done');
% end

%% POD-Galerkin projection

% Force compilation
diff_l( zeros(2,2,2,1,2,2) , 1 , 1);

if strcmp(param.type_data,'inc3D_HRLESlong_Re3900_blocks') ...
 ||  strcmp(param.type_data,'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks') ... 
 ||  strcmp(param.type_data,'DNS300_inc3d_3D_2017_04_09_NOT_BLURRED_blocks') 
%  || ( strcmp(param.type_data,'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks') ... 
%         && param.nb_modes > 2)...
%  || ( strcmp(param.type_data,'DNS300_inc3d_3D_2017_04_09_NOT_BLURRED_blocks') ... 
%         && param.nb_modes > 2)
    param.big_data = true;
end

% Compute the coefficients of the ODE of the Chronos b(t)
% I are the constant coefficients
% L are the linear coefficients
% C are the quadratic coefficients

% Coefficents in the deterministic case
[bool_exist,I_deter,L_deter,C_deter,param] ...
    = fct_all_file_save_1st_result(param);
if ~ bool_exist
    [I_deter,L_deter,C_deter,param] = ...
        param_ODE_bt_deter(param.name_file_mode, param, param.grid);
end
% if param.big_data
toc;tic;
disp('Galerkin projection on deterministic Navier-Stokes done');
% end
% Additional coefficients due to stochastic terms
[I_sto,L_sto,C_sto] = ...
    param_ODE_bt_sto(param.name_file_mode, param, param.grid);
I_sto = I_deter + I_sto;
L_sto = L_deter + L_sto;
C_sto = C_deter + C_sto;
% if param.big_data
toc;tic;
disp('Galerkin projection on stochastic Navier-Stokes done');
% end

%% Save first results, especially I, L, C
file_save = fct_file_save_1st_result(param);

save(file_save);
clear coef_correctif_estim
% if param.igrida
toc;tic;
disp('1st result saved');
% end

%% Time integration of the reconstructed Chronos b(t)

warning('loccal change of the time step');
n_mult_dt = 10;
if param.decor_by_subsampl.n_subsampl_decor >= 5
    n_mult_dt = n_mult_dt * param.decor_by_subsampl.n_subsampl_decor;
end
%n_mult_dt = 1;

bt_tot=bt_tot(1:(param.N_test+1),:); % reference Chronos

if isfield(param,'N_estim')
    bt_tronc=bt_tot(1:param.N_estim,:); % Initial condition
    param.N_test=param.N_test-param.N_estim+1;
else
    bt_tronc=bt_tot(1,:); % Initial condition
end

param.dt = param.dt /n_mult_dt;
param.N_test = param.N_test *n_mult_dt;

% Reconstruction in the deterministic case
bt_forecast_deter=bt_tronc;
for l = 1:param.N_test
    bt_forecast_deter= [bt_forecast_deter; ...
        evol_forward_bt_RK4(I_deter,L_deter,C_deter, param.dt, bt_forecast_deter)];
end

% Reconstruction in the stochastic case
bt_forecast_sto=bt_tronc;
for l = 1:param.N_test
    bt_forecast_sto = [bt_forecast_sto; ...
        evol_forward_bt_RK4(I_sto,L_sto,C_sto, param.dt, bt_forecast_sto) ];
end



param.dt = param.dt *n_mult_dt;
param.N_test = param.N_test /n_mult_dt;
bt_forecast_sto = bt_forecast_sto(1:n_mult_dt:end,:);
bt_forecast_deter = bt_forecast_deter(1:n_mult_dt:end,:);

clear bt_tronc
% if param.igrida
toc;
disp('Reconstruction of Chronos done');
% end

%% Save 2nd results, especially I, L, C and the reconstructed Chronos
if param.a_time_dependant
    dependance_on_time_of_a = '_a_time_dependant_';
else
    dependance_on_time_of_a = '_a_cst_';
end
if param.decor_by_subsampl.bool
    if strcmp(dependance_on_time_of_a,'a_t')
        char_filter = [ '_on_' param.type_filter_a ];
    else
        char_filter = [];
    end
    %         save([ param.folder_results '2ndresult_' param.type_data '_' num2str(param.nb_modes) '_modes_' ...
    %             dependance_on_time_of_a char_filter ...
    %             '_decor_by_subsampl_' num2str(param.decor_by_subsampl.meth) ...
    %             '_choice_' num2str(param.decor_by_subsampl.choice_n_subsample)  ...
    %             '_threshold_' num2str(param.decor_by_subsampl.spectrum_threshold) ...
    %             'fct_test_' param.decor_by_subsampl.test_fct '.mat']);
    file_save = [ param.folder_results '2ndresult_' param.type_data '_' num2str(param.nb_modes) '_modes_' ...
        dependance_on_time_of_a char_filter ...
        '_decor_by_subsampl_' num2str(param.decor_by_subsampl.meth) ...
        '_choice_' num2str(param.decor_by_subsampl.choice_n_subsample)  ...
        '_threshold_' num2str(param.decor_by_subsampl.spectrum_threshold) ...
        'fct_test_' param.decor_by_subsampl.test_fct];
else
    %     save([ param.folder_results '2ndresult_' param.type_data '_' num2str(param.nb_modes) '_modes_' ...
    %         dependance_on_time_of_a '.mat']);
    file_save=[ param.folder_results '2ndresult_' param.type_data '_' num2str(param.nb_modes) '_modes_' ...
        dependance_on_time_of_a];
end
if isfield(param,'N_estim')
    file_save=[file_save '_p_estim_' num2str(param.period_estim)];
end
file_save=[file_save '.mat'];
save(file_save);
clear C_deter C_sto L_deter L_sto I_deter I_sto
% if param.big_data
disp('2nd result saved');
% end

% Remove temporary files
if isfield(param,'folder_file_U_temp' )
    rmdir(param.folder_file_U_temp,'s');
end

%% Plots of the reconstructed Chronos

if plot_bts
    
    param.plot.plot_deter=plot_deterministic;
    param.plot.plot_EV=false;
    param.plot.plot_tuned=false;
    param.plot_modal_dt = false;
    
%     plot_bt_dB(param,bt_tot,bt_tot,...
%         bt_tot, bt_tot, bt_forecast_deter,...
%         bt_tot,bt_forecast_sto,bt_forecast_sto,bt_tot)
    
    if plot_each_mode
        plot_bt(param,bt_tot,bt_tot,...
            bt_tot, bt_tot, bt_forecast_deter,...
            bt_tot,bt_forecast_sto,bt_forecast_sto,bt_tot)
    end
    
    % if plot_bts
    %     plot_bt5(param,bt_forecast_sto,bt_forecast_deter,bt_tot)
    % end
end
