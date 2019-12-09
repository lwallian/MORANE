function super_main()
% Launch a set of simulations with a several set of parameters
% Especially several number of modes
%

init;
igrida=false;

%% Number of modes for the the ROM
% vect_nb_modes = 2 % For debugging
vect_nb_modes = [ 8 6 4 2] % For a full test
 % vect_nb_modes = 2.^(4:-1:1)
vect_reconstruction = [ false] % for the super_main_from_existing_ROM
% vect_adv_corrected = [ false]
vect_adv_corrected = [ true false]

%% Type of data
% Other datasets (do not use)
% % type_data = 'LES_3D_tot_sub_sample_blurred';
% % type_data = 'incompact3d_wake_episode3_cut_truncated';
% % type_data = 'inc3D_Re3900_blocks_truncated';
% % type_data = 'inc3D_Re3900_blocks';
% % type_data = 'incompact3D_noisy2D_40dt_subsampl';
% % type_data = 'incompact3D_noisy2D_40dt_subsampl_truncated';
% % type_data = 'inc3D_Re300_40dt_blocks_truncated';
% % type_data = 'turb2D_blocks_truncated'
% % type_data = 'test2D_blocks_truncated'
% % type_data = 'DNS300_inc3d_3D_2017_04_02_blocks_truncated'
% % type_data = 'inc3D_HRLESlong_Re3900_blocks_truncated'
% % type_data = 'small_test_in_blocks_truncated'
% % type_data = 'test_1_block'
% % type_data = 'inc3D_Re3900_blocks';
% % type_data = 'turb2D_blocks_truncated'

% These 3D data ( Re 300) gives good results
% type_data = 'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_truncated'

% These 2D data ( Re 100) gives good results
% type_data = 'DNS100_inc3d_2D_2018_11_16_blocks_truncated'

% Smaller dataset for debuging
type_data = 'incompact3D_noisy2D_40dt_subsampl_truncated'
% type_data = 'incompact3d_wake_episode3_cut_truncated'

%% Important parameters
switch type_data
        %  - Threshold used in the estimation of the optimal subsampling time step
        % - if modal-dt = true, 
        %   (mimic the use of a) disctinct subsampling time step for the
        %   differentials equations of distincts chronos
    case {'incompact3D_noisy2D_40dt_subsampl_truncated'}
        v_threshold=[1e-5]
        modal_dt=false
    case {'DNS100_inc3d_2D_2018_11_16_blocks_truncated'}
        % Threshold used in the estimation of the optimal subsampling time step
        v_threshold=1e-6 % BEST
        modal_dt=0
    case 'turb2D_blocks_truncated'
        v_threshold= [1e-5]
        modal_dt=0:1
    case {'incompact3d_wake_episode3_cut_truncated',...
            'incompact3d_wake_episode3_cut'}
        v_threshold=1e-6
        % %         v_threshold=1e-4
        modal_dt=false
        %         modal_dt=true;
    case {'LES_3D_tot_sub_sample_blurred',...
            'inc3D_Re3900_blocks',...
            'inc3D_Re3900_blocks_truncated'}
        v_threshold=1e-3
        modal_dt=true
    case 'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_truncated'
        v_threshold=1e-4 % BEST
        modal_dt=true;
    otherwise
        v_threshold=0.0005
        modal_dt=false
        modal_dt=0:2
end

%% Parameters which should not be modified
no_subampl_in_forecast = false;
% Estimation of corrective terms in the ROM
coef_correctif_estim.learn_coef_a=false;
% Time variation of the variance tensor
a_t = false;
% Saving all the reference chronos (not only resolved ones)
save_all_bi=false;
% (specific) time sub-sampling (forced time-decorrelation of unresolved chronos)
decor_by_subsampl.bool=true;
if (~ decor_by_subsampl.bool) || no_subampl_in_forecast
    error('There will be a problem in fct_name_2nd_result_new');
end
% Choice of subsampling time step based on chronos
decor_by_subsampl.test_fct='b';
% Way the subsampling is done (in which part of the code)
% (can be  'bt_decor' or  'a_estim_decor')
decor_by_subsampl.meth='bt_decor';
% Meth to choose the time sub-sampling
% ('auto_shanon'=maxim frequency of resolved chronos)
% ('lms' = correlation time estimation of the unresolved chronos through an lms filtered correlation function)
% ('truncated' = correlation time estimation of the unresolved chronos through a truncated correlation function)
% ('htgen' = correlation time estimation of the unresolved chronos through an heterogeneous estimator)
decor_by_subsampl.choice_n_subsample = 'htgen';
% 'auto_shanon' 'lms', 'truncated', 'htgen'
% Stochastic integration path: 'Ito' or 'Str'
global stochastic_integration;
stochastic_integration = 'Ito'

% Choose the correlated model (if true)
global correlated_model;
correlated_model = false

% Definition of global variable to manage methods more easily
global choice_n_subsample;
choice_n_subsample = decor_by_subsampl.choice_n_subsample;

% During the noise covariance estimation,
% remove the finite-variation part of the chronos
global estim_rmv_fv;
% estim_rmv_fv = false
estim_rmv_fv = true

if ~strcmp(choice_n_subsample, 'auto_shanon')
    v_threshold = NaN;
end
if correlated_model && strcmp(choice_n_subsample, 'auto_shanon')
    warning('Wrong choice of test function for correlated model. Switching to db')
    decor_by_subsampl.test_fct = 'db';
end

% Projection on the free-divergence-function space
% 0 : no projection / 1 : projection of deterministic terms
%  / 2 :projection of noise terms
eq_proj_div_free = 2

% Compute the variance tensor in the PIV space
% (if yes do not run the main code until the end)
global computed_PIV_variance_tensor
computed_PIV_variance_tensor = false
% DEFAULT : computed_PIV_variance_tensor = false

% Compute fake PIV snapshots
% (if yes do not run the main code until the end)
global compute_fake_PIV
compute_fake_PIV = false
% DEFAULT : compute_fake_PIV = false

% Compute PIV modes
% (if yes do not run the main code until the end)
global compute_PIV_modes
compute_PIV_modes = false
% DEFAULT : compute_PIV_modes = false

svd_pchol = true

%% Loops on chosen parameters
% The ROM is constructed and simulated on the learning basis
decor_by_subsampl.n_subsampl_decor=nan;

v_threshold_new = ones(length(vect_adv_corrected),1) * v_threshold ;
% v_threshold = ones(length(vect_adv_corrected),1) * v_threshold ;
vect_adv_corrected = vect_adv_corrected' * ones(1,length(v_threshold));
v_threshold = v_threshold_new; clear v_threshold_new
v_threshold = v_threshold(:)' ;
vect_adv_corrected = vect_adv_corrected(:)';
for q = 1:length(v_threshold)
    decor_by_subsampl_temp = decor_by_subsampl;
    threshold = v_threshold(q);
    decor_by_subsampl_temp.spectrum_threshold=threshold;
    adv_corrected = vect_adv_corrected(q);
    for k=vect_nb_modes
        k
        main(type_data,k,igrida,coef_correctif_estim,...
            save_all_bi,decor_by_subsampl_temp,a_t,adv_corrected,eq_proj_div_free);
    end
end

%% The ROM is simulated on a test basis
super_main_from_existing_ROM(...
    vect_nb_modes,type_data,v_threshold,modal_dt,...
    no_subampl_in_forecast,vect_reconstruction,vect_adv_corrected...
    ,decor_by_subsampl,eq_proj_div_free, svd_pchol)

