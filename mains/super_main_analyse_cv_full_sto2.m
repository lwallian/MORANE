function super_main_analyse_cv_full_sto2()
% Launch a set of simulations with a several set of parameters
% Especially several number of modes
%

init;
nb_modes_max=2;
% % N_particules=2;
% % n_simu=2;
% % N_particules=[1e5 1e4 1e3 1e2 1e1];
% % n_simu=[1e4 1e3 1e2 1e1];
% N_particules=[1e5 1e4 1e3 1e2 1e1];%1e1
N_particules=10.^[0:0.1:3];%1e1
n_simu=[1 25 50 75 100]; % 1e3
% n_simu=[1e4 1e3 1e2 1e1 1e1]; % 1e3
folder_results = [ pwd '/resultats/current_results/'];
current_pwd = pwd; cd ..
folder_data = [ pwd '/data/' ];
cd(current_pwd);
% % folder_results = ['/Users/Resseguier/Documents/MATLAB/POD/all/resultats/current_results/'];
% % folder_data = '/Users/Resseguier/Documents/MATLAB/POD/data/';
% type_data = 'incompact3d_wake_episode3_cut';
type_data = 'inc3D_Re3900_blocks_truncated';
var_bt_tot = nan(length(N_particules),length(n_simu));
for p=1:length(N_particules)
    for q=1:length(n_simu)
        load([ folder_results 'analyse_var_' type_data ...
            '_pcl_' N_particules(p) '_nsimu_' n_simu(q) '.mat'], ...
            'var_bt');
%             '_pcl_' num2str(N_particules(p)) '_nsimu_' num2str(n_simu(q)) '.mat'], ...
%             'var_bt');
%             '_pcl_' N_particules(p) '_nsimu_' n_simu(q) '.mat'], ...
%             'var_bt');
        var_bt_tot(p,q) =var_bt;
    end
end

%% save
save(['analyse_cv' type_data],'var_bt_tot');

%% plot
load(['analyse_cv' type_data],'var_bt_tot');
% load('analyse cv','var_bt_tot');
% figure;surf(log(var_bt_tot))
figure;surf((var_bt_tot(:,1:2)))
% figure;surf(log(var_bt_tot(:,1:3)))
figure;plot(var_bt_tot(:,3))
figure;plot(var_bt_tot(:,2))
 figure;plot(var_bt_tot(:,1))
% figure;plot(log(var_bt_tot(1,:)))
keyboard;