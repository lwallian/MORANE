function [eddy_visco,ILC] = estim_modal_non_lin_eddy_viscosity(bt,ILC,param)
% Estimate missing multiplicative coefficients in front of z_i by LS
% This coefficients can be different for different evolution equation (of
% different b_i)
%

% Get data
I_deter=ILC.deter.I;
L_deter=ILC.deter.L;
C_deter=ILC.deter.C;

[N,nb_modes]=size(bt);
nb_modes_used =param.coef_correctif_estim.nb_modes_used;

lambda=param.lambda;

if ~isnan(nb_modes_used)
    nb_modes_used=min(nb_modes,nb_modes_used);
    I_deter=I_deter(1:nb_modes_used);
    L_deter=L_deter(:,1:nb_modes_used);
    C_deter=C_deter(:,:,1:nb_modes_used);
else
    nb_modes_used=nb_modes;
end
if nb_modes_used~=nb_modes && ~( strcmp(type_estim,'scalar') || strcmp(type_estim,'vector_z') )
    error(['estimation impossible because the number of used modes is ' ...
        'reduced and the algorithm try to estimate a vector or a matrix']);
end
dt=param.dt;
N_learn_coef_a = param.N_learn_coef_a;
N_learn_coef_a = min(N_learn_coef_a,N-2);

% Numerical derivation
d_bt_obs = deriv_num_bt(bt(:,1:nb_modes_used),dt); % N x m

% Remove boundary pixels
d_bt_obs=d_bt_obs(2:(end-1),:);% (N-2) x m
bt=bt(2:(end-1),:);% (N-2) x m
bt=bt(1:N_learn_coef_a,:);% (N-2) x m
d_bt_obs=d_bt_obs(1:N_learn_coef_a,:);% (N-2) x m
[N,nb_modes]=size(bt);

d_b_deter = deriv_bt(I_deter,L_deter,C_deter, bt); % (N-2) x m
%%
% - I : m
% - L : m x m
% - C : m x m x m
% - bt : N x m

L_used = L_deter - param.C_deter_residu;
K=sum(lambda);
K_t = sum(bt.^2,2)/K;

X=zeros(nb_modes_used,N);
for i=1:nb_modes_used
    X(i,:) = sqrt(K_t').* (L_used(:,i)'*bt');
end

Y=(d_bt_obs - d_b_deter)';
X= -1/param.viscosity * X;
clear d_bt_obs d_b_deter

%% Least Square
eddy_visco=zeros(1,nb_modes);
for i=1:nb_modes
    Y_i = Y(i,:)';
    X_i = X(i,:)';
    eddy_visco(i)= LS_constrained(X_i,Y_i,-inf);
end

%%
    I_deter=ILC.deter.I;
    L_deter=ILC.deter.L;
    C_deter=ILC.deter.C;

%     eddy_visco : 1 x nb_modes
    
    ILC.NLMEV.I=I_deter;
    ILC.NLMEV.L=L_deter;
    ILC.NLMEV.L_EV=bsxfun(@times, eddy_visco/(param.viscosity*sqrt(K)) , L_used );
    ILC.NLMEV.C=C_deter;

end