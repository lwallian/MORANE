function [db_m, db_eta, db_Mi_ss, db_spiral] = evol_correlated_martingale_no_pchol(chol_xi_xi, chol_theta_theta, tau, dt, eta, spiral, Mi_ss, bt_m)
%EVOL_CORRELATED_MARTINGALE_NO_PCHOL FOR DEBUGGING PURPOSES ONLY.
% Evolves the correlated centered chronos using an Euler-Maruyama 
% integration scheme. Just martingale part without any auxiliary functions

[~ , n , nb_pcl ] = size(bt_m);

% Generate the noises
noises_xi = chol_xi_xi * randn(n, nb_pcl) * sqrt(dt);
noises_xi = permute(noises_xi, [1, 3, 4, 2]); % [n x 1 x 1 x nb_pcl]
noises_theta = chol_theta_theta * randn((n + 1) * n, nb_pcl) * sqrt(dt);
noises_theta = permute(noises_theta, [1, 3, 4, 2]); % [n * (n + 1) x 1 x 1 x nb_pcl]

% Evolve eta
db_sto_eta = reshape(noises_theta, [1, n + 1, n, nb_pcl]);
db_deter_eta = -eta / tau;
db_eta = eta + dt * db_deter_eta + db_sto_eta;

% Evolve spiral
db_sto_spiral = sqrt(dt) * sqrt(2 / tau) * randn(1, 1, nb_pcl);
db_deter_spiral = -spiral / tau;
db_spiral = spiral + dt * db_deter_spiral + db_sto_spiral;

% Evolve Mi_ss
db_deter_Mi = - 2 .* Mi_ss ./ tau;
db_sto_Mi = reshape(noises_xi, [1, n, nb_pcl]);
% db_sto_Mi = bsxfun(@times, db_spiral, noises_xi);
% db_sto_Mi = sum(db_sto_Mi, 3);
% db_sto_Mi = reshape(db_sto_Mi, [1, n, nb_pcl]);
db_Mi_ss = Mi_ss + dt * db_deter_Mi + db_sto_Mi;

clear noise_xi noise_theta;

% Evolve the equation with Euler-Maruyama
bt_x = permute(cat(2, bt_m, ones(1, 1, nb_pcl)), [1 2 4 3]);
eta_bt = bsxfun(@times, db_eta, bt_x);
eta_bt = sum(eta_bt, 2);
db_m = reshape(eta_bt, [1, n, nb_pcl]) + db_Mi_ss;

db_m = bt_m + db_m;

end
