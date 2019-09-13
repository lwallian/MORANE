function [bt_evol, b_fv, b_m, deta, dMi_ss, dspiral] = evol_correlated_no_pchol(I, L, C, ...
                        xi_xi, theta_theta, tau, dt, bt, eta, spiral, Mi_ss, bt_fv, bt_m)
%EVOL_CORRELATED_NO_PCHOL FOR DEBUGGING PURPOSES ONLY.
% Evolves the correlated centered chronos using an Euler-Maruyama 
% integration scheme.
%   @param I, L, C: deterministic model coefficients
%   @param xi_xi, theta_theta: noise coefficients for the stochastic part
%   @param tau: decay rate for the Ornstein-Uhlenbeck equations
%   @param dt: time step
%   @param bt: last chronos states
%   @param eta: last multiplicative noise coefficients
%   @param spiral: last noise evolution coefficient for Mi_ss
%   @return bt_evol: current chronos states
%   @return deta: current multiplicative noise coefficients
%   @return dMi_ss: current additive noise coefficients
%   @return dspiral: current noise evolution coefficient for Mi_ss

[~ , n , nb_pcl ] = size(bt);
[noise_xi, noise_theta] = generate_noises(xi_xi, theta_theta, n, nb_pcl, dt);
[deta, dMi_ss, dspiral] = evolve_sto_coeff(noise_xi, noise_theta, tau, eta, Mi_ss, spiral, n, nb_pcl, dt);
clear noise_xi noise_theta;

% Evolve the equation with Euler-Maruyama
db_m = evolve_sto(bt, dMi_ss, deta, n, nb_pcl, dt);
db_fv = evolve_deter(bt, I, L ,C, dt);
db_fv = permute(db_fv , [2 1 4 3]);

if nargin > 10
    b_fv = bt_fv + db_fv;
    b_m = bt_m + db_m;
%     bt_evol = db_fv + db_m;
% else
%     bt_evol = bt + db_fv + db_m;
end
bt_evol = bt + db_fv + db_m;

end


function db_fv = evolve_deter(bt, I, L, C, dt)

bt = permute(bt, [2 1 4 3]); % m x 1 x 1 x nb_pcl

C = bsxfun(@times, bt, C); % m x m x m x nb_pcl
C = permute(sum(C, 1), [2 3 1 4]); % m x m x 1 x nb_pcl


C = bsxfun(@times, bt, C); % m x m x 1 x nb_pcl
C = permute(sum(C, 1), [2 3 1 4]); % m x 1 x 1 x nb_pcl

L = bsxfun(@times, bt, L); % m x m x 1 x nb_pcl
L = permute(sum(L, 1), [2 3 1 4]); % m x 1 x 1 x nb_pcl

db_fv = - bsxfun(@plus, I, L + C ) * dt ; % m x 1 x 1

end


function db_m = evolve_sto(bt, Mi_ss, eta, n, nb_pcl, dt)

bt_x = permute(cat(2, bt, ones(1, 1, nb_pcl)), [1 2 4 3]);
eta_bt = bsxfun(@times, eta, bt_x);
eta_bt = sum(eta_bt, 2);
db_m = dt * reshape(eta_bt, [1, n, nb_pcl]) + dt * Mi_ss;

end


function [db_eta, db_Mi_ss, db_spiral] = evolve_sto_coeff(noise_xi, noise_theta, tau, eta, Mi_ss, spiral, n, nb_pcl, dt)

% Evolve both eta_i and M_i_ss
db_eta = evolve_eta(noise_theta, tau, eta, n, nb_pcl, dt);
[db_Mi_ss, db_spiral] = evolve_Mi_ss(noise_xi, tau, Mi_ss, spiral, n, nb_pcl, dt);

end


function db_eta = evolve_eta(noises, tau, eta, n, nb_pcl, dt)

% Evolve eta with Euler-Maruyama
db_sto = reshape(noises, [1, n + 1, n, nb_pcl]);
db_deter = -eta / tau;

db_eta = eta + dt * db_deter + db_sto;

end


function [db_Mi_ss, db_spiral] = evolve_Mi_ss(noises, tau, Mi_ss, spiral, n, nb_pcl, dt)

% Evolve Mi_ss with Euler-Maruyama
mi_ss_noise = noises;
db_spiral = evolve_spiral(spiral, tau, dt, nb_pcl);
db_deter = - 2 .* Mi_ss ./ tau;
% db_sto = mi_ss_noise; % for testing purposes
% warning('DEBUG');
db_sto = bsxfun(@times, db_spiral, mi_ss_noise);
db_sto = sum(db_sto, 3);
db_sto = reshape(db_sto, [1, n, nb_pcl]);

db_Mi_ss = Mi_ss + dt * db_deter + db_sto;

end

function db_spiral = evolve_spiral(spiral, tau, dt, nb_pcl)

% Evolve spiral with Euler-Maruyama
db_sto = sqrt(dt) * sqrt(2 / tau) * randn(1, 1, nb_pcl);
db_deter = -spiral / tau;

db_spiral = spiral + dt * db_deter + db_sto;

end


function [noises_xi, noises_theta] = generate_noises(xi_xi, theta_theta, n, nb_pcl, dt)

% Generate the noises matrix
noises_xi = xi_xi * randn(n, nb_pcl) * sqrt(dt);
noises_xi = permute(noises_xi, [1, 3, 4, 2]); % [n x 1 x 1 x nb_pcl]
noises_theta = theta_theta * randn((n + 1) * n, nb_pcl) * sqrt(dt);
noises_theta = permute(noises_theta, [1, 3, 4, 2]); % [n * (n + 1) x 1 x 1 x nb_pcl]
% noises = pchol_cov_noises * randn((n + 2) * n, nb_pcl) * sqrt(dt);
% noises = permute(noises, [1 3 4 2]); % (n + (n+1)*n) x 1 x 1 x nb_pcl

end
