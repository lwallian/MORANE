function super_main_full_sto_modal_dt()
% Launch a set of simulations with a several set of parameters
% Especially several number of modes
%

init;
% figure;
nb_modes_min = 26;
nb_modes_max=28;
% nb_modes_max=16;
% % nb_modes_max=2;

%% With correctif coefficient

% v_threshold=[1 10]/1000;
v_threshold=1e-4;
for q=1:length(v_threshold)
    for k=nb_modes_min:2:nb_modes_max
        main_full_sto_modal_dt(k,v_threshold(q))
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
    eval( ['print -depsc ' folder_results 'sum_modes_n=' ...
        num2str(nb_modes_max) '_threshold_' num2str(v_threshold(q)) '_fullsto_modal_dt.eps']);
end
