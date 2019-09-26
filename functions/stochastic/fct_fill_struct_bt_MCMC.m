function struct_bt_MCMC = fct_fill_struct_bt_MCMC(param,id_t,...
    struct_bt_MCMC,bt_MCMC_loc)
% Fill the structure struct_bt_MCMC
%

if length(size(bt_MCMC_loc))==3
    struct_bt_MCMC.tot.mean(id_t,:) = mean(bt_MCMC_loc, 3);
    struct_bt_MCMC.tot.var(id_t,:) = var(bt_MCMC_loc, 0, 3);
    struct_bt_MCMC.tot.one_realiz(id_t,:) = bt_MCMC_loc(:, :, 1);
    struct_bt_MCMC.qtl(id_t,:) = fx_quantile(bt_MCMC_loc, 0.025, 3);
    struct_bt_MCMC.diff(id_t,:) = fx_quantile(bt_MCMC_loc, 0.975, 3) - struct_bt_MCMC.qtl(id_t,:);
elseif length(size(bt_MCMC_loc))==4
    struct_bt_MCMC.tot.mean(id_t,:,:) = mean(bt_MCMC_loc, 4);
    struct_bt_MCMC.tot.var(id_t,:,:) = var(bt_MCMC_loc, 0, 4);
    struct_bt_MCMC.tot.one_realiz(id_t,:,:) = bt_MCMC_loc(:,:,:, 1);
    struct_bt_MCMC.qtl(id_t,:,:) = fx_quantile(bt_MCMC_loc, 0.025, 4);
    struct_bt_MCMC.diff(id_t,:,:) = fx_quantile(bt_MCMC_loc, 0.975, 4) - struct_bt_MCMC.qtl(id_t,:,:);    
else
    error('wrong size');
end
