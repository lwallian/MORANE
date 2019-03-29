function plot2b_evol(i,j,param)
% plot the evolution of b_i and b_j

%important parameters
type_data = param.type_data;
dt = param.dt;
folder_results = param.folder_results;
nb_modes = param.nb_modes;

%loading chronos
load([folder_results '2ndresult_' type_data '_' num2str(nb_modes) '_modes__a_cst__decor_by_subsampl_bt_decor_choice_auto_shanon_threshold_0_0005fct_test_b_fullsto.mat']);

bt_MCMC = permute(bt_MCMC, [3 1 2]);
bt_MCMC = mean(bt_MCMC);
bt_MCMC = permute(bt_MCMC, [2 3 1]);

b_i = bt_MCMC(:,i);
b_j = bt_MCMC(:,j);

t = dt*(0:size(b_i)-1);

plot3(b_i,b_j,t);
xlabel(['b_{' num2str(i) '}'])
ylabel(['b_{' num2str(j) '}'])
zlabel('time')
end
