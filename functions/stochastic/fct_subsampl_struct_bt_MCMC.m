function struct_bt_MCMC = fct_subsampl_struct_bt_MCMC(param,n_simu,...
    struct_bt_MCMC)
% Fill the structure struct_bt_MCMC
%

siz = size(struct_bt_MCMC.tot.mean);
struct_bt_MCMC.tot.mean = ...
    fct_resh_subsmpl(siz,n_simu,struct_bt_MCMC.tot.mean);
struct_bt_MCMC.tot.var = ...
    fct_resh_subsmpl(siz,n_simu,struct_bt_MCMC.tot.var);
struct_bt_MCMC.tot.one_realiz = ...
    fct_resh_subsmpl(siz,n_simu,struct_bt_MCMC.tot.one_realiz);
struct_bt_MCMC.qtl = ...
    fct_resh_subsmpl(siz,n_simu,struct_bt_MCMC.qtl);
struct_bt_MCMC.diff = ...
    fct_resh_subsmpl(siz,n_simu,struct_bt_MCMC.diff);

function field = fct_resh_subsmpl(siz_,n_simu_,field)
    field = reshape(field,[siz_(1) prod(siz_(2:end)) ] );
    field = field(1 : n_simu_ : end, :);
    field = reshape(field,[size(field,1) siz_(2:end) ] );
end

end