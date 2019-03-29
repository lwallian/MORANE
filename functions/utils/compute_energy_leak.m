% Computing the energy leak
clear all;
close all;

type_data = 'inc3D_Re300_40dt_blocks_test_basis';
% folder_data = 'F:\GitHub/data/';
nb_modes = 8;

% We suppose that the results have been already generated
% folder_results = 'F:\GitHub\PODFS/resultats/current_results/';
folder_results = [ pwd '/resultats/current_results/'];
current_pwd = pwd; cd ..
folder_data = [ pwd '/data/' ];
cd(current_pwd);

param = read_data_blocks(type_data, folder_data);

% Open the file containing all info on bi
switch type_data
    case 'inc3D_Re300_40dt_blocks_test_basis'
        % Load file containing chronos
        full_path = folder_results + "2ndresult_DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_truncated_"+num2str(nb_modes)+"_modes__a_cst__decor_by_subsampl_bt_decor_choice_auto_shanon_threshold_0.0001fct_test_b_fullsto_modal_dt.mat";
        load(full_path);
        folder_data_in_blocks = 'data_test_in_blocks/';
        file_prefix = 'inc40Dt_';
        
        % Load file containing topos
        full_path = folder_data + "mode_" + type_data(1:end-10) + "_truncated_"+num2str(nb_modes)+"_modes.mat";
        load(full_path);
end

% Computing the energy of temporal modes
% e2 = squeeze(sum(permute(bt_MCMC.^2, [2 1 3])));

% Important parameters
n_subsampl = param.decor_by_subsampl.n_subsampl_decor;
MX = param.MX;
N_particules = size(bt_MCMC,3);
e_moy = zeros(1,size(bt_MCMC,1));

% Computes energy for each particule (TODO : vectorize the function)
for n=1:N_particules
    figure();
    e = zeros(1,size(bt_MCMC,1));
    for t=1:size(bt_MCMC,1)
        e_temp = phi_m_U(:,end,:);
        for m=1:size(bt_MCMC,2)
            e_temp = e_temp + bt_MCMC(t,m,n)*phi_m_U(:,m,:);
        end
        e(t) = sum(sum(e_temp.^2))/(2*MX(1)*MX(2));
    end
    plot(e)
    title("Resolved energy for particule number " + num2str(n))
    xlabel("Time")
    ylabel("Resolved energy")
    e_moy = e_moy + e;
end

figure();
e_moy = e_moy/N_particules;
plot(e)
title("Mean resolved energy for all particules")
xlabel("Time")
ylabel("Resolved energy")