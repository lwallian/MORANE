function super_main_full_sto_modal_dt_2nd_result()
% Launch a set of simulations with a several set of parameters
% Especially several number of modes
%

init;
% % figure;
% nb_modes_min=2;
% nb_modes_max=32;
% % nb_modes_max=2;

% nb_modes_min=2;
% nb_modes_max=6;
% % % nb_modes_min=28;
% % % nb_modes_max=32;
% % vect_nb_modes = [8 16];
% % vect_nb_modes = nb_modes_min:2:nb_modes_max;
% vect_nb_modes = 2.^5;
vect_nb_modes = 2.^(1:4);
% vect_nb_modes = 2.^(1:5);
nb_modes_max = max(vect_nb_modes);


% Type of data

% % % These 3D data give good results
% % % They are saved in only one file
% % % (~ 250 time step)
% % % type_data = 'LES_3D_tot_sub_sample_blurred';
% % % type_data = 'incompact3d_wake_episode3_cut';
% % %     type_data = 'incompact3D_noisy2D_40dt_subsampl'; 
% % type_data = 'inc3D_Re3900_blocks';
% type_data = 'incompact3d_wake_episode3_cut';
type_data = 'incompact3d_wake_episode3_cut_truncated';
% type_data = 'inc3D_Re3900_blocks_truncated';

%% With correctif coefficient

% v_threshold=[1 10]/1000;
switch type_data
    case {'incompact3d_wake_episode3_cut',...
            'incompact3d_wake_episode3_cut_truncated'}
        v_threshold=0.0005
        modal_dt = false;
    case {'inc3D_Re3900_blocks',...
            'inc3D_Re3900_blocks_truncated'}
        v_threshold=1e-4
        modal_dt = true;
end

for q=1:length(v_threshold)
    for k=vect_nb_modes
%     for k=nb_modes_min:2:nb_modes_max
        %         main_full_sto_modal_dt(k,v_threshold(q))
        main_full_sto_modal_dt_2nd_res(type_data,k,v_threshold(q))
    end
    %% Save plot
    folder_results = [ pwd '/resultats/current_results/'];
    current_pwd = pwd; cd ..
    folder_data = [ pwd '/data/' ];
    cd(current_pwd);
    % folder_results = ['/Users/Resseguier/Documents/MATLAB/POD/all/resultats/current_results/'];
    % folder_data = '/Users/Resseguier/Documents/MATLAB/POD/data/';
    % folder_results = ['/Users/vressegu/Documents/matlab/POD-NS_Stochastique/current_used/' ...
    %         'all/resultats/current_results/'];
    % %     param.folder_results =  [ pwd '/resultats/current_results/'];
    %     eval( ['print -depsc ' folder_results 'sum_modes_n=' num2str(nb_modes_max) ...
    %             '_threshold_' num2str(v_threshold(q)) '.eps']);
    
    threshold = num2str(v_threshold(q));
    iii = (threshold =='.');
    threshold(iii)='_';
    
    str = ['print -depsc ' folder_results type_data '_sum_modes_n=' ...
        num2str(nb_modes_max) '_threshold_' threshold ...
        '_fullsto'];
    if modal_dt
        str =[ str '_modal_dt'];
    end
    str =[ str '.eps'];
    eval(str);

%     eval( ['print -depsc ' folder_results type_data '_sum_modes_n=' ...
%         num2str(nb_modes_max) '_threshold_' threshold ...
%         '_fullsto_modal_dt.eps']);
end
