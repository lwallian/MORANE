function super_main_analyse_cv()
% Launch a set of simulations with a several set of parameters
% Especially several number of modes
%

init;
% nb_modes_min = 2;
% nb_modes_max=10;
% % % figure;
% % nb_modes_min=2;
% % nb_modes_max=6;
% % % % nb_modes_min=28;
% % % % nb_modes_max=32;
% % % vect_nb_modes = [8 16];
% % % vect_nb_modes = nb_modes_min:2:nb_modes_max;
vect_nb_modes = 2.^(1:4);
nb_modes_max = max(vect_nb_modes);


% % % N_particules=2;
% % % n_simu=2;
% % % N_particules=[1e5 1e4 1e3 1e2 1e1];
% % % n_simu=[1e4 1e3 1e2 1e1];
% % N_particules=[1e5 1e4 1e3 1e2 1e1];%1e1

N_particules=[1 10 50 100 200];%1e1
% N_particules=ceil(10.^[0:0.5:3]);%1e1

% N_particules=10.^[0];%1e1
n_simu=100;
% n_simu=[1e4 1e3 1e2 1e1 1e1]; % 1e3
folder_results = [ pwd '/resultats/current_results/'];
current_pwd = pwd; cd ..
folder_data = [ pwd '/data/' ];
cd(current_pwd);
% % folder_results = ['/Users/Resseguier/Documents/MATLAB/POD/all/resultats/current_results/'];
% % folder_data = '/Users/Resseguier/Documents/MATLAB/POD/data/';
% type_data = 'incompact3d_wake_episode3_cut';

% type_data = 'inc3D_Re3900_blocks_truncated';
type_data = 'incompact3d_wake_episode3_cut_truncated';

for nb_modes=vect_nb_modes
%     for nb_modes=nb_modes_min:2:nb_modes_max
    
    var_bt = nan(length(N_particules),length(n_simu));
    for p=1:length(N_particules)
%         for q=1:length(n_simu)
            fprintf('N_particules')
            N_particules(p)
            var_bt(p) = analyse_cv(type_data,nb_modes,N_particules(p),n_simu);
            %         var_bt(p,q) = analyse_cv_full_sto(nb_modes_max,N_particules(p),n_simu(q));
%         end
    end
    
    %% save
    save([ folder_results 'analyse_cv' type_data '_' num2str(nb_modes)],...
        'N_particules','var_bt');
    % save('analyse cv','var_bt');
    N_particules_p=reshape(N_particules,size(var_bt));
    plot(N_particules_p,var_bt)
%     surf(var_bt)
    
end



for nb_modes=vect_nb_modes
%     for nb_modes=nb_modes_min:2:nb_modes_max
    
    
    load([ folder_results 'analyse_cv' type_data '_' num2str(nb_modes)]);
    figure;semilogx(N_particules,var_bt,'o-')
    xlabel('Nb particules');
    title([num2str(nb_modes) ' modes']);
%     xlabel('$\log_{10}(\text{Nb particules})$','interpreter','latex')
    
%     var_bt_tot = nan(length(N_particules),1);
%     for p=1:length(N_particules)
%         %for q=1:length(n_simu)
%         load([ folder_results 'analyse_var_' type_data ...
%             '_pcl_' N_particules(p) '.mat'], ...
%             'var_bt');
%         var_bt_tot(p) =var_bt;
%         %end
%     end
%     
%     %% save
%     save(['analyse_cv' type_data],'var_bt_tot');
%     
%     %% plot
%     load(['analyse_cv' type_data],'var_bt_tot');
%     % load('analyse cv','var_bt_tot');
%     % figure;surf(log(var_bt_tot))
%     figure;plot(var_bt_tot)
%     % figure;surf(log(var_bt_tot(:,1:3)))
%     figure;plot(var_bt_tot(:,3))
%     figure;plot(var_bt_tot(:,2))
%     figure;plot(var_bt_tot(:,1))
%     % figure;plot(log(var_bt_tot(1,:)))
%    keyboard;
end