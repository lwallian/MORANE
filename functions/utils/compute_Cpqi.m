function e = compute_Cpqi(param, p, q, i)
% computes non-linear energy transfers from p q modes to i mode (?)

%important parameters
type_data = param.type_data;
folder_results = param.folder_results;
nb_modes = param.nb_modes;

%loading chronos
load([folder_results '2ndresult_' type_data '_' num2str(nb_modes) '_modes__a_cst__decor_by_subsampl_bt_decor_choice_auto_shanon_threshold_0.0005fct_test_b_fullsto.mat']);
bibpbq = bt_MCMC(:,i,:).*bt_MCMC(:,p,:).*bt_MCMC(:,q,:);
bibpbq = permute(bibpbq, [1 3 2]);
bibpbq = mean(bibpbq, 2);

e = C_deter(p,q,i)*bibpbq;
end