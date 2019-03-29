function super_main_full_sto_modal_dt_2_test_basis_2D(...
    vect_nb_modes,type_data,v_threshold,modal_dt)
% Launch a set of simulations with a several set of parameters
% Especially several number of modes
%


% Number of periods reconstructed
nb_period_test = 9;% for DNS 300
% nb_period_test = 5;%for DNS 3900

if nargin == 0
    init;
    % % figure;
    % nb_modes_min=2;
    % nb_modes_max=6;
    % % % nb_modes_min=28;
    % % % nb_modes_max=32;
    % % vect_nb_modes = [8 16];
    % % vect_nb_modes = nb_modes_min:2:nb_modes_max;
    % vect_nb_modes = 2.^5;
    vect_nb_modes = 2.^(1:4);
    
    %% Type of data
    % These 3D data give good results
    % They are saved in only one file
    % (~ 250 time step)
    % type_data = 'LES_3D_tot_sub_sample_blurred';
    type_data = 'incompact3d_wake_episode3_cut_truncated';
    % type_data = 'inc3D_Re3900_blocks_truncated';
    % type_data = 'inc3D_Re3900_blocks';
    %     type_data = 'incompact3D_noisy2D_40dt_subsampl';
    
    % These 3D data are bigger, since the spatial grid is thinner
    % and the number of time step is bigger
    % (~ 2000 time step)
    % They are saved in different files
    % The internship should try to use these data
    %     type_data = 'inc3D_Re3900_blocks';
    
    
    %% With correctif coefficient
    
    % v_threshold=[1 10]/1000;
    % v_threshold=1e-3;
    if strcmp(type_data,'incompact3d_wake_episode3_cut_truncated')
        v_threshold=0.0005
        modal_dt=false;
    else
        v_threshold=1e-4
        modal_dt=true;
    end
    
end

nb_modes_max = max(vect_nb_modes);


for q=1:length(v_threshold)
    for k=vect_nb_modes
        %     for k=nb_modes_min:2:nb_modes_max
        main_full_sto_modal_dt_test_basis(k,v_threshold(q),type_data,nb_period_test)
        %         main_full_sto_modal_dt_2nd_res(k,v_threshold(q))
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
    %         num2str(nb_modes_max) '_threshold_' num2str(v_threshold(q)) ...
    %         '_fullsto_modal_dt.eps']);
    % %     eval( ['print -depsc ' folder_results 'sum_modes_n=' ...
    % %         num2str(nb_modes_max) '_threshold_' num2str(v_threshold(q)) '_fullsto_modal_dt_test_basis.eps']);
end
