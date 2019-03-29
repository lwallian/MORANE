function super_main_analyse_cv_full_sto()
% Launch a set of simulations with a several set of parameters
% Especially several number of modes
%

init;
% nb_modes_max=2;
nb_modes_min = 8;
nb_modes_max=8;
% % N_particules=2;
% % n_simu=2;
% N_particules=[1e5 1e4 1e3 1e2 1e1];
% n_simu=[1e4 1e3 1e2 1e1 1e1];


% % % N_particules=2;
% % % n_simu=2;
% % % N_particules=[1e5 1e4 1e3 1e2 1e1];
% % % n_simu=[1e4 1e3 1e2 1e1];
% % N_particules=[1e5 1e4 1e3 1e2 1e1];%1e1
% N_particules=10.^[0:0.5:3];%1e1
N_particules=10.^[0];%1e1
n_simu=[50 75]; % 1e3
% n_simu=[1e4 1e3 1e2 1e1 1e1]; % 1e3
folder_results = [ pwd '/resultats/current_results/'];
current_pwd = pwd; cd ..
folder_data = [ pwd '/data/' ];
cd(current_pwd);
% % folder_results = ['/Users/Resseguier/Documents/MATLAB/POD/all/resultats/current_results/'];
% % folder_data = '/Users/Resseguier/Documents/MATLAB/POD/data/';
% type_data = 'incompact3d_wake_episode3_cut';
type_data = 'inc3D_Re3900_blocks_truncated';

for nb_modes=nb_modes_min:2:nb_modes_max
    
    var_bt = nan(length(N_particules),length(n_simu));
    for p=1:length(N_particules)
        for q=1:length(n_simu)
            fprintf('N_particules')
            N_particules(p)
            fprintf('n_simu')
            n_simu(q)
            var_bt(p,q) = analyse_cv_full_sto_2(type_data,nb_modes,N_particules(p),n_simu(q));
            %         var_bt(p,q) = analyse_cv_full_sto(nb_modes_max,N_particules(p),n_simu(q));
        end
    end
    
    %% save
    save([ folder_results 'analyse_cv' type_data '_' num2str(nb_modes)],'var_bt');
    % save('analyse cv','var_bt');
    surf(var_bt)
    
end