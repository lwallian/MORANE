function super_main_EV()
% Launch a set of simulations with a several set of parameters
% Especially several number of modes
%


init;

%% Number of modes for the the ROM
% vect_nb_modes = 2 % For debugging
vect_nb_modes = [ 16 8 6 4 2] % For a full test
% % vect_nb_modes = 2.^(4:-1:1)

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

% Data of the Re 100 OpenFOAM simulation
% type_data = 'DNS100_OpenFOAM_2D_2020_blocks_truncated'
% type_data = 'DNS100_OpenFOAM_2D_2020_blocks'
% type_data = 'DNS100_OpenFOAM_2D_2020'

% These 3D data ( Re 300) gives good results
% type_data = 'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_truncated'

% These 2D data ( Re 100) gives good results
% type_data = 'DNS100_inc3d_2D_2018_11_16_blocks_truncated'

% Smaller dataset for debuging
type_data = 'incompact3D_noisy2D_40dt_subsampl_truncated'
% % type_data = 'incompact3d_wake_episode3_cut_truncated'

% Additive noise ?
vect_add_noise = [ false true];

% Adapt time differntiation and time scale
rigorous_EV_noise_estim = true;

%% With correctif coefficient
% for q=1:length(v_threshold)

for add_noise = vect_add_noise
    for k=vect_nb_modes
        % for k=2:2:nb_modes_max
        main_EV(k,type_data,add_noise,rigorous_EV_noise_estim);
    end
end

