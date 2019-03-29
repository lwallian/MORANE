function super_main_modal_dt()
% Launch a set of simulations with a several set of parameters
% Especially several number of modes
%

init;
figure;
%v_nb_modes=2
% v_nb_modes=2.^(1:2)
 v_nb_modes=2.^(1:4)
% % nb_modes_max=16;
% % % nb_modes_max=2;
nb_modes_max=max(v_nb_modes);

%%

% Type of data

% These 3D data give good results
% They are saved in only one file
% (~ 250 time step)
type_data = 'LES_3D_tot_sub_sample_blurred';
type_data = 'incompact3d_wake_episode3_cut';
type_data = 'inc3D_Re3900_blocks';
type_data = 'inc3D_HRLESlong_Re3900_blocks';
%type_data = 'DNS300_inc3d_3D_2017_04_09_NOT_BLURRED_blocks'
%type_data = 'DNS300_inc3d_3D_2017_04_09_blocks'
% type_data = 'DNS100_inc3d_2D_2017_04_29_blocks'

%%

% Threshold of the Chonos spectrum, used to choice the time step
threshold=0.01; % for HRLESlong 3900
threshold=0.0001 ;% for HRLESlong 3900
% %         v_threshold= [5e-4 4e-4 1e-4] % % for 'DNS300_inc3d_3D_2017_04_09_NOT_BLURRED_blocks'
threshold= 5e-4; % pas assez de diffusion
% threshold= 4e-4; % pas assez de diffusion
%threshold= 1e-4; % pas assez de diffusion
% threshold= 1e-2; % pas assez de diffusion
% threshold= 1e-3 % pas assez de diffusion
%threshold=1.5e-4 %

% threshold=0.001; % 0.001 or 0.01 for LES 3900 (or inc3D 3900)
% threshold=0.005; % for LES 3900
% threshold=0.0005; % for inc3D episode 3
% threshold=0.00014; % for inc3D 3900
% threshold=0.000135; % for inc3D 3900

% v_threshold= [1e-2 1e-3 ] %
%  v_threshold= [1e-2 1e-3 1e-4  1e-5] %
%  %v_threshold= [5e-4 1e-4  1e-5] %
% %v_threshold= [5e-4 1e-4  1e-5] %
  v_threshold= 1e-4 % BEST for Re 100
  v_threshold= 1e-5 % BEST for Re 3900
%  warning('NOT BEST THRESHOLD');

%% With correctif coefficient
for q=1:length(v_threshold)
    
    for k=v_nb_modes
        % for k=2:2:nb_modes_max
        param = main_modal_dt(k,type_data,v_threshold(q));
    end
    
    %% Save plot
    % folder_results = ['/Users/vressegu/Documents/matlab/POD-NS_Stochastique/current_used/' ...
    %         'all/resultats/current_results/'];
    folder_results =  [ pwd '/resultats/current_results/'];
    eval( ['print -depsc ' folder_results 'sum_modes_n=' num2str(nb_modes_max) ...
        '_' type_data ...
        '_threshold_' num2str(v_threshold(q)) '.eps']);
    close all
    
end
