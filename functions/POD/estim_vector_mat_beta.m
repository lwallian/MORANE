function [beta,ILC] = estim_vector_mat_beta(bt,ILC,param)
% Estimate missing multiplicative coefficients in front of z_i by LS
% This coefficients can be different for different evolution equation (of
% different b_i)
%
type_estim=param.coef_correctif_estim.type_estim;
beta_min=param.coef_correctif_estim.beta_min;

% Get data
I_deter=ILC.deter.I;
L_deter=ILC.deter.L;
C_deter=ILC.deter.C;
I_sto=ILC.sto.I;
L_sto=ILC.sto.L;
C_sto=ILC.sto.C;

[N,nb_modes]=size(bt);
nb_modes_used =param.coef_correctif_estim.nb_modes_used;

lambda=param.lambda;

if ~isnan(nb_modes_used)
    nb_modes_used=min(nb_modes,nb_modes_used);
    I_deter=I_deter(1:nb_modes_used);
    L_deter=L_deter(:,1:nb_modes_used);
    C_deter=C_deter(:,:,1:nb_modes_used);
    I_sto=I_sto(1:nb_modes_used);
    L_sto=L_sto(:,1:nb_modes_used);
    C_sto=C_sto(:,:,1:nb_modes_used);
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
bt = permute(bt,[2 3 1]); % m x 1 x N

X=zeros(nb_modes_used,nb_modes+1,N);
for i=1:nb_modes_used
    X(i,1,:) = multiprod(multitrans(bt) , repmat(L_sto(:,i), [1 1 N]));
end
for q=1:nb_modes
    for i=1:nb_modes_used
        X(i,q+1,:) = multiprod(multitrans(bt) , repmat(C_sto(:,q,i), [1 1 N])) ...
            .*bt(q,1,:) - repmat( lambda(q)*C_sto(q,q,i) , [1 1 N]);
    end
end

Y=(d_bt_obs - d_b_deter)';
X=-X;
clear d_bt_obs d_b_deter

%% Least Square

switch type_estim
    case 'scalar'
        X=sum(X,2);
        X=X(:);
        Y=Y(:);
        beta= LS_constrained(X,Y,0);
    case 'vector_z'
        X=permute(X,[1 3 2]);% nb_modes x N x nb_modes+1
        X=reshape(X,[nb_modes_used*N, nb_modes+1]);
        Y=Y(:);
        beta= LS_constrained(X,Y,beta_min);
        
        beta=beta';
    case 'vector_b'
        X=sum(X,2);% nb_modes x 1 x N
        beta=zeros(nb_modes,1);
        for i=1:nb_modes
            Y_i = Y(i,:)';
            X_i = permute(X(i,1,:),[3 1 2]);
            beta(i)= LS_constrained(X_i,Y_i,beta_min);
        end
    case 'matrix'
        beta=zeros(nb_modes,nb_modes+1)';
        for i=1:nb_modes
            Y_i = Y(i,:)';
            X_i = squeeze(X(i,:,:))';
            beta(:,i)= LS_constrained(X_i,Y_i,beta_min);
        end
        beta=beta';
end

%%
if nargout >=2
    I_deter=ILC.deter.I;
    L_deter=ILC.deter.L;
    C_deter=ILC.deter.C;
    if strcmp(type_estim,'scalar')
        I_sto=beta*ILC.sto.I;
        L_sto=beta*ILC.sto.L;
        C_sto=beta*ILC.sto.C;
    else
        I_sto=ILC.sto.I;
        L_sto=ILC.sto.L;
        C_sto=ILC.sto.C;
        
        if size(beta,2)>1
            C_sto= bsxfun( @times,C_sto,permute(beta(:,2:end),[3 2 1]));
        else
            C_sto= bsxfun( @times,C_sto,permute(beta,[3 2 1]));
        end
        L_sto= bsxfun( @times,L_sto,beta(:,1)');
        for q=1:nb_modes
            I_sto(q)=-trace(diag(lambda)*C_sto(:,:,q));
        end
    end
    ILC.sto.I=I_sto+I_deter;
    ILC.sto.L=L_sto+L_deter;
    ILC.sto.C=C_sto+C_deter;
end

end