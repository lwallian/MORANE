% Plot the energy and variance over time
clear all;
close all;

%% Parameters of the program
% type_data = 'DNS300_inc3d_3D_2017_04_02_blocks';
% type_data = 'turb2D_blocks';
param.type_data = 'DNS100_inc3d_2D_2018_11_16_blocks_truncated'

switch param.type_data
    case {'DNS100_inc3d_2D_2018_11_16_blocks_truncated'}
        param.decor_by_subsampl.spectrum_threshold = 1e-6
        param.nb_modes = 8
        param.adv_corrected = true
%         param.adv_corrected = false
        modal_dt = 1
    case 'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_truncated'
        param.decor_by_subsampl.spectrum_threshold = 1e-4
        param.nb_modes = 8
        param.adv_corrected = false
        modal_dt = 1
    otherwise
        error('unknown case')
end
reconstruction = false;
param.a_time_dependant = false;
param.decor_by_subsampl.bool=true;
param.decor_by_subsampl.test_fct='b';
param.decor_by_subsampl.meth='bt_decor';
param.decor_by_subsampl.choice_n_subsample='auto_shanon';
param.decor_by_subsampl.no_subampl_in_forecast = false;

% folder_results = 'F:\GitHub\PODFS/resultats/current_results/';
param.folder_results = [ pwd '/resultats/current_results/'];
current_pwd = pwd; cd ..
param.folder_data = [ pwd '/data/' ];
cd(current_pwd);

%% Loading the correct results file and important parameters included in it

param = fct_name_2nd_result(param,modal_dt,reconstruction);
load(param.name_file_2nd_result)
% results_file = [folder_results '2ndresult_' type_data '_truncated_' num2str(nb_modes) ...
%     '_modes__a_cst__decor_by_subsampl_bt_decor_choice_auto_shanon_threshold_' ...
%     threshold 'fct_test_b_fullsto_modal_dt.mat'];
% load(results_file)
% % dt = 3600*24;
clearvars -except bt_MCMC type_data folder_results nb_modes current_pwd
[N_t, nb_modes, N_particules] = size(bt_MCMC);

%% Loop on modes
for i=1:nb_modes
    fig = figure;
    bi = bt_MCMC(:,i,:);
    bi = permute(bi, [ 1 3 2 ]);
    he = animatedline('Color','r', 'DisplayName', 'Expected value');
    hv = animatedline('Color','b', 'DisplayName', 'Variance');
    T = linspace(0,1,N_t);
    % Loop on time
    for t=1:N_t
        bit = bi(t,:)';
        e = mean(bit);
        v = var(bit);
        addpoints(he,T(t),e);
        drawnow
        addpoints(hv,T(t),v);
        drawnow
    end
    
end

