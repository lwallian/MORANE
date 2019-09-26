%% Test to verify that the correlated model is indeed calculating correctly
% each of the noise terms. ILC and noise matrices generated synthetically
% and the reestimated to make sure its working correctly

%% Test the function for estimating the correlated model's noises
clear all, close all, clc;
% s = RandStream('mt19937ar','Seed',0);
% Random generator
rng('default'); %  The default settings are the Mersenne Twister with seed 0.

% Define simulation parameters
n = 2; % number of modes
% tau_stab = 100;
tau_stab = 10;
% Temps_integ = 800
Temps_integ = 8 % 8 80 800
% % dt = 0.00008
% dt = 0.0008
% dt = 0.01
dt = 0.001
T = floor(Temps_integ / dt) % number of simulation steps
n_particles = 10
time = 0 : dt : dt * (T - 1);

% Load the determinisitic ROM parameters
% file_test = [ pwd '\resultats\current_results\' ...
%     '1stresult_DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_truncated' ...
%     '_2_modes_htgenb_fullsto\_no_correct_drift_integ_Ito.mat']
% load(file_test, 'ILC','bt_tot');
load('ILC_test.mat', 'ILC','bt_tot');


bt = zeros(T, n, n_particles);
bt(1, :, :) = repmat(bt_tot(1, :), [1, 1, n_particles]);

I = ILC.tot.I;
L = ILC.tot.L;
C = ILC.tot.C;

% I = 0;L=0;C=0;

L = L + (1/tau_stab) * eye(n);

% Generate some ROM parameters randomly
coef_expl = 1e-2
tau_min = 24 / 100
% tau_min = 24 / 10
tau = tau_min

chol_theta_theta = ...
    ( randn( [n+1,n,n+1,n] ) / sqrt(n * (n)) );
chol_theta_theta(1:n,:,:,:) = chol_theta_theta(1:n,:,:,:) * ...
    sqrt( coef_expl * (1/(tau*tau_min^2)) ) ;
chol_theta_theta(n+1,:,:,:) = chol_theta_theta(n+1,:,:,:) * ...
    sqrt( ( mean(bt(1, :, 1) / tau)).^2 / tau_min ) ;
chol_theta_theta = reshape( chol_theta_theta , [n * (n + 1),n * (n + 1)]);
% chol_theta_theta = sqrt( coef_expl * (1/(tau*tau_min^2)) ) * ...
%     ( randn(n * (n + 1)) / sqrt(n * (n + 1)) );
% % chol_theta_theta = sqrt( coef_expl * (n/(tau*tau_min^2)) ) * ...
% %     ( randn(n * (n + 1)) / (n * (n + 1)) );
% % chol_theta_theta = sqrt( coef_expl * (n/(tau*tau_min^2)) ) * randn(n * (n + 1));
theta_theta =  chol_theta_theta * chol_theta_theta' ;
theta_theta = 0.5 * (theta_theta + theta_theta');
[V, D] = eig(theta_theta);
D = diag(D);
D(D < 0) = 0;
D = diag(D);
theta_theta = V * D * V';
chol_theta_theta = V * sqrt(D);

% chol_xi_xi = randn(n);
chol_xi_xi = sqrt( (2 * mean(bt(1, :, 1) / tau)).^2 / tau_min ) ...
    * (randn(n)/sqrt(n));
% chol_xi_xi = sqrt( (2 * mean(bt(1, :, :) / tau)).^2 / tau_min ) * randn(n);
xi_xi =  chol_xi_xi * chol_xi_xi' ;
trace(xi_xi)

% tau_xi_xi = (2 * mean(bt(1, :, :) / tau)).^2 / trace(xi_xi)

xi_xi = 0.5 .* (xi_xi + xi_xi');
[V, D] = eig(xi_xi);
D = diag(D);
D(D < 0) = 0;
D = diag(D);
xi_xi = V * D * V';
chol_xi_xi = V * sqrt(D);

tau_xi_xi = (2 * mean(bt(1, :, 1) / tau)).^2 / (trace(xi_xi)/n)

theta_xi = zeros(n * (n + 1), n);

result1 = [xi_xi; theta_xi];
result2 = [theta_xi'; theta_theta];
result = [result1, result2];

result = 0.5 .* (result + result');
[V, D] = eig(result);
D = diag(D);
D(D < 0) = 0;
D = diag(D);
result = V * D * V';
chol_result = V * sqrt(D);




clear result1 result2;

bt_fv = zeros(T, n, n_particles);
bt_m = zeros(T, n, n_particles);
eta = zeros(T, n + 1, n, n_particles);
Mi_ss = zeros(T, n, n_particles);
spiral = zeros(T, 1, n_particles);

% warning('DEBUG');
% spiral = ones(T, 1, n_particles);
%% Integrate the chronos
% for p=1:n_particles
%     for k = 1 : T - 1
%         [bt(k + 1, :, p), eta(k + 1, :, :, p) ...
%             , Mi_ss(k + 1, :, p), spiral(k + 1,:, p)] = ...
%             evol_forward_correlated_centered(I, L, C, ...
%             chol_result, tau, dt, bt(k, :, p), eta(k, :, :, p),...
%             spiral(k, :,  p), Mi_ss(k, :, p));
%         %     [bt(k + 1, :, :), bt_fv(k + 1, :, :), bt_m(k + 1, :, :), eta(k + 1, :, :, :) ...
%         %         , Mi_ss(k + 1, :, :), spiral(k + 1, : ,:, :)] = ...
%         %         evol_forward_correlated_centered(I, L, C, ...
%         %         chol_result, tau, dt, bt(k, :, :), eta(k, :, :, :),...
%         %         spiral(k, :, :, :), Mi_ss(k, :, :), bt_fv(k, :, :), bt_m(k, :, :));
%         % %     [bt(k + 1, :, :), bt_fv(k + 1, :, :), bt_m(k + 1, :, :), eta(k + 1, :, :, :) ...
%         % %         , Mi_ss(k + 1, :, :), spiral(k + 1, : ,:, :)] = ...
%         % %         evol_correlated_no_pchol(I, L, C, ...
%         % %         chol_xi_xi, chol_theta_theta, tau, dt, bt(k, :, :), eta(k, :, :, :),...
%         % %         spiral(k, :, :, :), Mi_ss(k, :, :), bt_fv(k, :, :), bt_m(k, :, :));
%     end
% end

% Integrate the chronos
for k = 1 : T - 1
    [bt(k + 1, :, :), eta(k + 1, :, :, :) ...
        , Mi_ss(k + 1, :, :), spiral(k + 1, : ,:, :)] = ...
        evol_forward_correlated_centered(I, L, C, ...
        chol_result, tau, dt, bt(k, :, :), eta(k, :, :, :),...
        spiral(k, :, :, :), Mi_ss(k, :, :));
%     [bt(k + 1, :, :), bt_fv(k + 1, :, :), bt_m(k + 1, :, :), eta(k + 1, :, :, :) ...
%         , Mi_ss(k + 1, :, :), spiral(k + 1, : ,:, :)] = ...
%         evol_forward_correlated_centered(I, L, C, ...
%         chol_result, tau, dt, bt(k, :, :), eta(k, :, :, :),...
%         spiral(k, :, :, :), Mi_ss(k, :, :), bt_fv(k, :, :), bt_m(k, :, :));
% %     [bt(k + 1, :, :), bt_fv(k + 1, :, :), bt_m(k + 1, :, :), eta(k + 1, :, :, :) ...
% %         , Mi_ss(k + 1, :, :), spiral(k + 1, : ,:, :)] = ...
% %         evol_correlated_no_pchol(I, L, C, ...
% %         chol_xi_xi, chol_theta_theta, tau, dt, bt(k, :, :), eta(k, :, :, :),...
% %         spiral(k, :, :, :), Mi_ss(k, :, :), bt_fv(k, :, :), bt_m(k, :, :));
end

%% Plot the resulting solutions
% figure(1), plot(time,mean(bt_fv, 3)), title('Finite variation'), grid minor;
% figure(2), plot(time,mean(bt_m, 3)), title('Martingale'), grid minor;
figure(3), plot(time,mean(bt, 3)), title('Total'), grid minor;

% Plot the different stochastic mean variables
% mean_spiral = mean(spiral, 3);
% mean_Mi_ss = mean(Mi_ss, 3);
% mean_eta = mean(eta, 4);
% figure(4), plot(time,mean_spiral), title('spiral'), grid minor;
% figure(5), plot(time,mean_Mi_ss), title('Mi_{ss}'), grid minor;
% figure(6), plot(time,reshape(mean_eta(:,end,:), T, [])), grid minor, title('eta(end,:)')
% figure(7), plot(time,reshape(mean_eta(:,1:(end-1),:), T, [])), grid minor, title('eta(1:(end-1),:)')
% figure(8), plot(time,1./reshape(mean_eta(:,1:(end-1),:), T, [])), grid minor, title('1/eta(1:(end-1),:)')
% ax=axis;ax(3:4)=ax(3:4)*1e-4;axis(ax);
% % figure(6), plot(time,reshape(mean_eta(:,2,1), T, [])), grid minor, title('eta 21')
% % figure(7), plot(time,reshape(mean_eta(:,1,2), T, [])), grid minor, title('eta 12')
% % figure(16), plot(time,reshape(1./mean_eta(:,2,1), T, [])), grid minor, title('1/eta 21')
% % ax=axis;ax(3:4)=ax(3:4)*1e-3;axis(ax);
% % % axis([time(1) time(end) -10 10]);
% % figure(17), plot(time,reshape(1./mean_eta(:,1,2), T, [])), grid minor, title('1/eta 12')
% % ax=axis;ax(3:4)=ax(3:4)*1e-3;axis(ax);
% % % axis([time(1) time(end) -10 10]);

eta2 = sum(sum(eta(:,1:end-1,:,1).^2,3),2);
tau_eta = 1./sqrt( mean(eta2 ,1) ) 

%%
result_est = nan([size(result) n_particles]);
pseudo_chol_est = nan([size(result) n_particles]);
for p=1:n_particles
    % Estimate the parameters with the corresponding function
    [result_est(:,:,p), pseudo_chol_est(:,:,p)] = estimate_noise_non_orthogonal(...
        bt(:,:,p), eta(:,:,:,p), n, T, dt, 1, false);
end
% [result_est, pseudo_chol_est] = estimate_noise_non_orthogonal(...
%     bt, eta, n, T, dt, n_particles, false);
% % [theta_theta_est, xi_xi_est] = estimate_noise_matrices_orthogonal(...
% %     bt, eta, n, T, dt, n_particles, false);

% % Estimate the error made while estimating
% err_result = (result - result_est).^2 / mean((result(:)).^2);
% sq_mean_err_result = sqrt(mean(err_result(:)))

% err_pseudo_chol = (pseudo_chol - pseudo_chol_est).^2 / mean((pseudo_chol(:)).^2);
% sq_mean_err_pseudo_chol = sqrt(mean(err_pseudo_chol(:)))

% Estimate the error made while estimating
theta_theta_est = result_est((n+1):end,(n+1):end,:);
err_theta = (theta_theta - theta_theta_est).^2 / mean((theta_theta(:)).^2);
sq_mean_err_theta = sqrt(mean(err_theta(:)))
% on_sq_N_ech = 1/sqrt( (T*dt)/tau )

xi_xi_est = result_est(1:n,1:n,:);
err_xi_xi = (xi_xi - xi_xi_est).^2 / mean((xi_xi(:)).^2);
sq_mean_err_xi_xi = sqrt(mean(err_xi_xi(:)))
% on_sq_N_ech = 1/sqrt( (T*dt)/tau )

on_sq_N_ech = 1/sqrt( (T*dt)/tau )
