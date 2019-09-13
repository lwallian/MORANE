%% Synthetic test for the noise terms' estimation formulas
clear all, close all, clc;
s = RandStream('mt19937ar','Seed',0);
% Random generator
rng('default'); %  The default settings are the Mersenne Twister with seed 0.

% Define some hyperparameters
n = 2; % number of modes
% Temps_integ = 80
Temps_integ = 80
% T = 100000; % number of simulation steps
% T = 10000; % number of simulation steps
% T_cov = floor(T / 2);
% dt = 1e-7; % dt = 0.08;
dt = 0.0008;
% dt = 0.008;
T = floor(Temps_integ/dt) % number of simulation steps
n_particles = 1;
time = 0 : dt : dt * (T - 1);

load('ILC_test.mat', 'ILC','bt_tot');
load('noises_test.mat', 'xi_xi_inf', 'theta_theta');

% Generate some ROM parameters randomly
coef_expl = 1e-2
tau_min = 24/10
% tau_min = 3e3 * dt
tau = tau_min
% % tau = 1e3 * dt
% % tau = 20 * dt
% alpha_d = 1 / tau;
% % alpha_m = 1e-6;

I = ILC.tot.I;
% I = alpha_d .* ones(n, 1);
L = ILC.tot.L;
% L = alpha_d .* ones(n, n);
C = ILC.tot.C;
% C = alpha_d.^2 .* ones(n, n, n);

% I = zeros(n, 1);
% L = zeros(n, n);
% C = zeros(n, n, n);

chol_theta_theta = sqrt( coef_expl * (n/(tau*tau_min^2)) ) * randn(n * (n + 1));
% chol_theta_theta = sqrt( coef_expl * (n/tau^3) ) * randn(n * (n + 1));

% chol_theta_theta = zeros(size(chol_theta_theta));

theta_theta =  chol_theta_theta * chol_theta_theta' ;

% theta_theta = (n/tau^3) .* eye(n * (n + 1));
% theta_theta = tau / n * 1000 .* eye(n * (n + 1));

% theta_theta(n + 1, n + 1) = 0; theta_theta(end, end) = 0;

% theta_theta = reshape(theta_theta, [n + 1, n, n + 1, n]);
% theta_theta_c = theta_theta;
% for p = 1 : n
%     for i = 1 : n
%         for q = 1 : n
%             for j = 1 : n
%                 theta_theta(p, i, q, j) = 0.5 * (theta_theta_c(p, i, q, j) - theta_theta_c(i, p, q, j));
%             end
%         end
%     end
% end
% theta_theta_c = theta_theta;
% for p = 1 : n
%     for i = 1 : n
%         for q = 1 : n
%             for j = 1 : n
%                 theta_theta(p, i, q, j) = 0.5 * (theta_theta_c(p, i, q, j) - theta_theta_c(p, i, j, q));
%             end
%         end
%     end
% end
% theta_theta = reshape(theta_theta, [n * (n + 1), n * (n + 1)]);


% % theta_theta = zeros((n + 1) * n, (n + 1) * n);
theta_theta = reshape(theta_theta, [n * (n + 1), n * (n + 1)]);
% % theta_theta = alpha_m .* rand((n + 1) * n, (n + 1) * n);
% theta_theta = 0.5 .* (theta_theta + theta_theta');
% % theta_theta = theta_theta + n * (n + 1) * alpha_m * eye(n * (n + 1));
theta_theta = 0.5 * (theta_theta + theta_theta');
[V, D] = eig(theta_theta);
D = diag(D);
D(D < 0) = 0;
D = diag(D);
theta_theta = V * D * V';
chol_theta_theta = V * sqrt(D);
% % chol_theta_theta = chol(theta_theta);

bt = zeros(T, n, n_particles);
bt(1, :, :) = repmat(bt_tot(1, :), [1, 1, n_particles]);
% bt(1,:,:) = repmat(bt_tot(end,:), [1, 1, n_particles]);

% xi_xi = zeros(n, n);
% xi_xi = 50 * xi_xi_inf;
% xi_xi = xi_xi_inf /30;
% xi_xi = xi_xi_inf /3;
chol_xi_xi = sqrt( (2 * mean(bt(1, :, :) / tau)).^2 / tau_min ) * randn(n);
xi_xi =  chol_xi_xi * chol_xi_xi' ;

% xi_xi = alpha_m .* rand(n, n);
xi_xi = 0.5 .* (xi_xi + xi_xi');
% xi_xi = xi_xi + n * alpha_m * eye(n);
[V, D] = eig(xi_xi);
D = diag(D);
D(D < 0) = 0;
D = diag(D);
xi_xi = V * D * V';
chol_xi_xi = V * sqrt(D);
theta_xi = zeros(n * (n + 1), n);

% Start the chronos at zero
% bt = zeros(T, n, n_particles);

% bt(1,1,:)=1;
% bt(1,2,:)=0;

% bt(1,:,:) = repmat(bt_tot(end,:), [1, 1, n_particles]);

bt_fv = zeros(T, n, n_particles);
bt_m = zeros(T, n, n_particles);
% bt_m(1, :, :) = randn(n, n_particles);
eta = zeros(T, n + 1, n, n_particles);
% eta(1, :, :, :) = randn(1, n + 1, n, n_particles);
Mi_ss = zeros(T, n, n_particles);
% Mi_ss(1, :, :) = randn(1, 1, n, n_particles);
% spiral = zeros(T, 1, n_particles);
% spiral(1, :, :) = randn(1, 1, n_particles);
spiral = ones(T, 1, n_particles);

% Simulate the evolution of the Chronos equations

for k = 1 : T - 1
    [bt(k + 1, :, :), bt_fv(k + 1, :, :), bt_m(k + 1, :, :), eta(k + 1, :, :, :) ...
        , Mi_ss(k + 1, :, :), spiral(k + 1, : ,:, :)] = ...
        evol_correlated_no_pchol(I, L, C, ...
        chol_xi_xi, chol_theta_theta, tau, dt, bt(k, :, :), eta(k, :, :, :),...
        spiral(k, :, :, :), Mi_ss(k, :, :), bt_fv(k, :, :), bt_m(k, :, :));
    
%     figure(3), plot(time(1:k),mean(bt(1:k,:,:), 3)), title('Total'), grid minor;
%     mean_eta = mean(eta, 4);
%     figure(6), plot(time(1:k),reshape(dt*mean_eta(1:k,2,1), k, [])), grid minor, title('dt * eta 21')

end

% Plot the resulting solutions
% figure(1), plot(time,mean(bt_fv, 3)), title('Finite variation'), grid minor;
% figure(2), plot(time,mean(bt_m, 3)), title('Martingale'), grid minor;
figure(3), plot(time,mean(bt, 3)), title('Total'), grid minor;

% Plot the different stochastic mean variables
mean_spiral = mean(spiral, 3);
mean_Mi_ss = mean(Mi_ss, 3);
mean_eta = mean(eta, 4);
figure(4), plot(time,mean_spiral), title('spiral'), grid minor;
figure(5), plot(time,mean_Mi_ss), title('Mi_{ss}'), grid minor;
% figure(6), plot(time,reshape(mean_eta(:,2,1), T, [])), grid minor, title('eta 21')
% figure(7), plot(time,reshape(mean_eta(:,1,2), T, [])), grid minor, title('eta 12')
% figure(16), plot(time,reshape(1./mean_eta(:,2,1), T, [])), grid minor, title('1/eta 21')
% ax=axis;ax(3:4)=ax(3:4)*1e-3;axis(ax);
% % axis([time(1) time(end) -10 10]);
% figure(17), plot(time,reshape(1./mean_eta(:,1,2), T, [])), grid minor, title('1/eta 12')
% ax=axis;ax(3:4)=ax(3:4)*1e-3;axis(ax);
% % axis([time(1) time(end) -10 10]);

eta2 = sum(sum(eta(:,1:end-1,:).^2,3),2);
tau_eta = 1./sqrt( mean(eta2 ,1) ) 
% tau_eta = 1./sqrt(mean( mean_eta(:,1,2).^2 )) 

% t = 0 : dt : dt * (T_cov - 1);

% Verify that eta is effectively a frequency
% eta_t = mean_eta(:, 1 : end - 1, :);
% eta_t = permute(eta_t, [2 3 1]);
% eig_eta = zeros(T, n);
% for k = 1 : T
%     eig_eta(k, :) = eig(eta_t(:, :, k));
% end
% any(isreal(eig_eta))
% 
% % Plot the modes in the complex plane
% figure(8), hold on, plot(eig_eta(:, 1), 'or'), plot(eig_eta(:, 2), 'xb'), grid minor, hold off;

% Do the pertinent estimations
diff_bt = (bt(2:end,:,:)-bt(1:end-1,:,:))/(dt);
diff_bt(end,:,:,:)=[];
% diff_bt = (bt(3:end,:,:)-bt(1:end-2,:,:))/(2*dt);
eta(1,:,:,:)=[];
eta(end,:,:,:)=[];
bt(1,:,:)=[];
bt(end,:,:)=[];
T=T-2;
d2bt = (diff_bt(2:end,:,:)-diff_bt(1:end-1,:,:))/(dt);
d2bt_c = d2bt - mean(d2bt, 1);
% d2bt = diff(diff_bt, 1, 1)/(dt);
% d2bt = diff(diff(bt, 1, 1), 1, 1)/(dt^2);
% d2bt = diff(bt, 2, 1)/(dt^2);
deta = (eta(2:end,:,:)-eta(1:end-1,:,:))/(dt);
deta_c = deta - mean(deta, 1);
% deta = diff(eta, 1, 1)/(dt);
% deta(1,:,:,:)=[];
% deta(end,:,:,:)=[];
% bt(1:2,:,:)=[];
% bt(end-1:end,:,:)=[];
% d2bt = diff(bt, 2, 1);
% deta = diff(eta, 1, 1);
bt_x = cat(2, bt, ones(T, 1, n_particles));
G_pq = zeros(n + 1, n + 1, n_particles);
for k = 1 : T
    for p = 1 : n + 1
        for q = 1 : n + 1
            G_pq(p, q, :) = G_pq(p, q, :) + bt_x(k, p, :) .* bt_x(k, q, :);
        end
    end
end
G_pq = G_pq ./ T
% G_pq = bt_x' * bt_x ./ T;

%%

% Beta
beta = zeros(n + 1, n, n + 1, n, n_particles);
for l = 1 : n_particles
    for p = 1 : n + 1
        for i = 1 : n
            for q = 1 : n + 1
                for j = 1 : n
                    for k = 1 : T
                        beta(p, i, q, j, l) = beta(p, i, q, j, l) + ...
                            bt_x(k, p, l) * d2bt_c(k, i, l) * deta_c(k, q, j, l);
%                             bt_x(k, p, l) * d2bt(k, i, l) * deta(k, q, j, l);
%                             bt_x(k, p, l) * d2bt(k, i, l) * deta(k, q, j, l) / G_pq(p, p, l);
                    end
                end
            end
        end
    end
end
beta = beta .* dt ./ T

% theta_theta_est
theta_theta_est = zeros(n + 1, n, n + 1, n, n_particles);
for k = 1 : n_particles
    for i = 1 : n
        for q = 1 : n + 1
            for j = 1 : n
                weights_inv = diag(sqrt(diag(G_pq(:, :, k))).^(-1));
                weighted_G_pq = weights_inv * G_pq(:, :, k) * weights_inv;
                weighted_beta = weights_inv * beta(:, i, q, j, k);
                theta_theta_est(:, i, q, j, k) = weighted_G_pq \ weighted_beta;
                theta_theta_est(:, i, q, j, k) = weights_inv * theta_theta_est(:, i, q, j, k);
%                 theta_theta_est_a(:, i, q, j, k) = G_pq(:, :, k) \ beta(:, i, q, j, k);
%                 theta_theta_est(:, i, q, j, k) = linsolve(G_pq(:, :, k), beta(:, i, q, j, k));
%                 theta_theta_est(:, i, q, j, k) = pinv_G * beta(:, i, q, j);
            end
        end
    end
end

%%


% % Beta
% theta_theta_est = zeros(n + 1, n, n + 1, n, n_particles);
% for l = 1 : n_particles
%     for p = 1 : n + 1
%         for i = 1 : n
%             for q = 1 : n + 1
%                 for j = 1 : n
%                     for k = 1 : T - 2
%                         theta_theta_est(p, i, q, j, l) = theta_theta_est(p, i, q, j, l) + ...
%                             bt_x(k, p, l) * d2bt(k, i, l) * deta(k, q, j, l) / G_pq(p, p, l);
%                     end
%                 end
%             end
%         end
%     end
% end
% theta_theta_est = theta_theta_est .* dt ./ (T - 2);
% warning('DEBUG')

%%

% xi_xi_est
xi_xi_est = zeros([size(xi_xi), n_particles]);
for k = 1 : n_particles
    for i = 1 : n
        for j = 1 : n
            kappa = d2bt_c(:, i, k)' * d2bt_c(:, j, k) * dt / (T - 2);
%             kappa = d2bt(:, i, k)' * d2bt(:, j, k) * dt / (T - 2);
            gamma = 0;
            for p = 1 : n + 1
                for q = 1 : n + 1
                    gamma = gamma + G_pq(p, q, k) * theta_theta_est(p, i, q, j, k);
                end
            end
            xi_xi_est(i, j, k) = kappa - gamma;
%             xi_xi_est(i, j, k) = kappa_a - gamma;
        end
    end
end

theta_theta_est = reshape(theta_theta_est, [n * (n + 1), n * (n + 1), n_particles]);
% theta_theta_est_a = reshape(theta_theta_est_a, [n * (n + 1), n * (n + 1), n_particles]);

theta_theta_est = 0.5* ( theta_theta_est + theta_theta_est');
[U,S,~]=svd(theta_theta_est);
S=diag(S);S(S<0)=0;S=diag(S);
theta_theta_est = U * S * U';

xi_xi_mean = mean(xi_xi_est, 3);
% theta_theta_mean = mean(reshape(theta_theta_est, [n * (n + 1), n * (n + 1), n_particles]), 3);

%%

theta_theta
theta_theta_est
% theta_theta_est_a

err_theta = (theta_theta - theta_theta_est).^2 / mean((theta_theta(:)).^2)
% err_theta_a = (theta_theta - theta_theta_est_a).^2 / mean((theta_theta(:)).^2)
sq_mean_err_theta = sqrt(mean(err_theta(:)))
% sq_mean_err_theta_a = sqrt(mean(err_theta_a(:)))

on_sq_N_ech = 1/sqrt( (T*dt)/tau )

%%

xi_xi
xi_xi_est

err_xi_xi = (xi_xi - xi_xi_est).^2 / mean((xi_xi(:)).^2)
sq_mean_err_xi_xi = sqrt(mean(err_xi_xi(:)))

on_sq_N_ech = 1/sqrt( (T*dt)/tau )

% xi_xi_est
% xi_xi