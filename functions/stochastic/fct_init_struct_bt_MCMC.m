function struct_bt_MCMC = fct_init_struct_bt_MCMC(param,bt_MCMC_loc)
% Initialize the structure struct_bt_MCMC
%

siz = size(bt_MCMC_loc);
siz=siz(2:end-1);

struct_bt_MCMC.tot.mean=nan([param.N_test+1 siz ]);
struct_bt_MCMC.tot.var=nan([param.N_test+1 siz ]);
struct_bt_MCMC.tot.one_realiz=nan([param.N_test+1 siz ]);
struct_bt_MCMC.qtl=nan([param.N_test+1 siz ]);
struct_bt_MCMC.diff=nan([param.N_test+1 siz ]);

struct_bt_MCMC = fct_fill_struct_bt_MCMC(param,1,...
    struct_bt_MCMC,bt_MCMC_loc);
