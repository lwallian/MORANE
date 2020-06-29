function [eddy_visco,ILC] = estim_eddy_viscosity(bt,ILC,param)
% Estimate missing multiplicative coefficients in front of z_i by LS
% This coefficients can be different for different evolution equation (of
% different b_i)
%
% type_estim=param.coef_correctif_estim.type_estim;
% eddy_visco_min=param.coef_correctif_estim.eddy_visco_min;
% beta_min=-inf;

% Get data
I_deter=ILC.deter.I;
L_deter=ILC.deter.L;
C_deter=ILC.deter.C;

[N,nb_modes]=size(bt);
% nb_modes_used =nb_modes;
% nb_modes_used =2;
nb_modes_used =param.coef_correctif_estim.nb_modes_used;

lambda=param.lambda;
% %% Test
% for q=1:nb_modes
%     ecart(q)=abs(I_sto(q)+trace(diag(lambda)*C_sto(:,:,q)))/abs(I_sto(q));
% end
% ecart
% keyboard;

if ~isnan(nb_modes_used)
    nb_modes_used=min(nb_modes,nb_modes_used);
    %     bt_used=bt(:,1:nb_modes_used);
    I_deter=I_deter(1:nb_modes_used);
    L_deter=L_deter(:,1:nb_modes_used);
    C_deter=C_deter(:,:,1:nb_modes_used);
else
    nb_modes_used=nb_modes;
    %     bt_used=bt;
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
bt_ini = bt;
bt=bt(2:(end-1),:);% (N-2) x m
bt=bt(1:N_learn_coef_a,:);% (N-2) x m
d_bt_obs=d_bt_obs(1:N_learn_coef_a,:);% (N-2) x m
[N,nb_modes]=size(bt);

d_b_deter = deriv_bt(I_deter,L_deter,C_deter, bt); % (N-2) x m

L_used = L_deter - param.C_deter_residu;

X=zeros(nb_modes_used,N);
for i=1:nb_modes_used
    X(i,:) = L_used(:,i)'*bt';
end

Y=(d_bt_obs - d_b_deter)';
X= -1/param.viscosity * X;
clear d_b_deter

%% Least Square
% eddy_visco=zeros(1,nb_modes);
for i=1:nb_modes
    Y_i = Y(i,:)';
    X_i = X(i,:)';
%     eddy_visco(i)= LS_constrained(X_i,Y_i,-inf);
end
X=X(:);
Y=Y(:);
eddy_visco = (X'*X)\(X'*Y);
eddy_visco = max([0 eddy_visco]);

I_deter=ILC.deter.I;
L_deter=ILC.deter.L;
C_deter=ILC.deter.C;

ILC.EV.I=I_deter;
ILC.EV.L=L_deter + eddy_visco/param.viscosity * L_used ;
% ILC.MEV.L=L_deter + bsxfun(@times, eddy_visco/param.viscosity , L_used );
ILC.EV.C=C_deter;

%% Noise
% if param.add_noise
%     err = Y  - eddy_visco * X;
%     sigma_err = sqrt( mean(err(:).^2) * dt );
%     ILC.EV.sigma_err = sigma_err;
% end
if param.add_noise    
%     sigma_err=zeros(1,nb_modes);
%     for i=1:nb_modes
%         Y_i = Y(i,:)';
%         X_i = X(i,:)';
%         err_i = Y_i  - eddy_visco * X_i;
%         sigma_err(i) = sqrt( mean(err_i(:).^2) * dt );
%     end
%     ILC.EV.sigma_err = sigma_err;
    
    d_b_deter = deriv_bt(ILC.EV.I,ILC.EV.L,ILC.EV.C, bt); % (N-2) x m
    % Error
    
    if param.rigorous_EV_noise_estim
        % Time diff
        d_bt_obs = bt_ini(3:end,1:nb_modes_used) ...
            - bt_ini(2:end-1,1:nb_modes_used);
        d_bt_obs = d_bt_obs / dt ;
    end

    err = (d_bt_obs - d_b_deter)';
    % Remove bias
    err = bsxfun(@plus, err, - mean(err,2));
    mat_cov_error = 1/(size(err,2)-1) * (err * err') * dt  ;
    
    if param.rigorous_EV_noise_estim
        c_prime = 1/(size(err,1)) * (err' * err);
        correlation = estimateCovS(c_prime);
        % correlation = estimateAutocorrelation(cov_s);
        dt_correlation = diff(correlation);
        % dt_correlation = diff(correlation)./ dt;
        dt_correlation = [ 0; dt_correlation ];
        % The time derivative is always 0 at delta t = 0 because of the symetry
        tau = sqrt(2 * mean(correlation.^2) / mean( dt_correlation.^2));
        % tau = sqrt(2 * mean((correlation).^2) / mean((diff(correlation) ./ dt).^2));
        % tau = tau / dt;
        mat_cov_error = tau * mat_cov_error;
    end
    
    p_chol = zeros([param.nb_modes*(param.nb_modes+1),param.nb_modes]);
    p_chol(1:param.nb_modes,:) = chol(mat_cov_error,'lower');
    ILC.EV.pchol_cov_noises = p_chol;
end

end