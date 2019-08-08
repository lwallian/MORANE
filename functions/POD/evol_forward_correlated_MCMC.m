function [bt_evol, db_fv, db_m, eta, Mi_ss, Gr] = evol_forward_correlated_MCMC(I, L, C, ...
                        pchol_cov_noises, tau, dt, bt, eta, Gr, Mi_ss, Mi_sigma, bt_fv, bt_m)
%EVOL_FORWARD_CORRELATED_MCMC Evolves the correlated velocity model using
%an Euler-Maruyama integration scheme
%   @param I, L, C: deterministic model coefficients
%   @param pchol_cov_noises: noise coefficients for the stochastic part
%   @param tau: decay rate for the Ornstein-Uhlenbeck equations
%   @param dt: time step
%   @param bt: last chronos states
%   @param eta: last multiplicative noise coefficients
%   @param Gr: last noise evolution coefficient for Mi_ss
%   @param Mi_ss: last additive noise coefficients
%   @return bt_evol: current chronos states
%   @return eta: current multiplicative noise coefficients
%   @return Mi_ss: current additive noise coefficients
%   @return Gr: current noise evolution coefficient for Mi_ss

[~ , n , nb_pcl ] = size(bt);
noises = generate_noises(pchol_cov_noises, n, nb_pcl, dt);
[eta, Mi_ss, Gr] = evolve_sto_coeff(noises, pchol_cov_noises, Mi_sigma, tau, eta, Mi_ss, Gr, n, nb_pcl, dt);
clear noises;

% Evolve the equation with Euler-Maruyama
db_m = evolve_sto(bt, Mi_ss, eta, n, nb_pcl);
db_fv = evolve_deter(bt, I, L ,C, dt);
db_fv = permute(db_fv , [2 1 4 3]);

if nargin > 11
    db_fv = bt_fv + db_fv;
    db_m = bt_m + db_m;
end

bt_evol = bt + db_fv + db_m;

end


function db_fv = evolve_deter(bt, I, L, C, dt)

bt = permute(bt,[2 1 4 3]); % m x 1 x 1 x nb_pcl

C = bsxfun(@times,bt,C); % m x m x m x nb_pcl
C = permute(sum(C,1),[2 3 1 4]); % m x m x 1 x nb_pcl


C = bsxfun(@times,bt,C); % m x m x 1 x nb_pcl
C = permute(sum(C,1),[2 3 1 4]); % m x 1 x 1 x nb_pcl

L = bsxfun(@times,bt,L); % m x m x 1 x nb_pcl
L = permute(sum(L,1),[2 3 1 4]); % m x 1 x 1 x nb_pcl

db_fv = - bsxfun(@plus, I, L + C ) * dt ; % m x 1 x 1

end


function db_m = evolve_sto(bt, Mi_ss, eta, n, nb_pcl)

bt_x = permute(cat(2, bt, ones(1, 1, nb_pcl)), [1 2 4 3]);
eta_bt = bsxfun(@times, eta, bt_x);
eta_bt = sum(eta_bt, 2);
db_m = reshape(eta_bt, [1, n, nb_pcl]) + Mi_ss;

end


function [db_eta, db_Mi_ss, db_Gr] = evolve_sto_coeff(noises, pchol_cov_noises, Mi_sigma, tau, eta, Mi_ss, Gr, n, nb_pcl, dt)

% Evolve both eta_i and M_i_ss
db_eta = evolve_eta(noises, tau, eta, n, nb_pcl, dt);
[db_Mi_ss, db_Gr] = evolve_Mi_ss(pchol_cov_noises, tau, Mi_sigma, Mi_ss, Gr, n, nb_pcl, dt);

end


function db_eta = evolve_eta(noises, tau, eta, n, nb_pcl, dt)

% Evolve eta with Euler-Maruyama
db_sto = reshape(noises(n + 1 : end, 1, 1, :), [1, n + 1, n, nb_pcl]);
db_deter = -eta / tau;

db_eta = eta + dt * db_deter + db_sto;

end


function [db_Mi_ss, db_Gr] = evolve_Mi_ss(pchol_cov_noises, tau, Mi_sigma, Mi_ss, Gr, n, nb_pcl, dt)

% Evolve Mi_ss with Euler-Maruyama
mi_ss_noise = randn(1, 1, n, nb_pcl) .* sqrt(dt);
db_Gr = evolve_Gr(pchol_cov_noises, Gr, tau, dt, nb_pcl, n);
db_deter = - 2 * Mi_ss / tau + reshape(repmat(Mi_sigma, [1, 1, nb_pcl]), size(Mi_ss));
db_sto = bsxfun(@times, db_Gr, mi_ss_noise);
db_sto = sum(db_sto, 3);
db_sto = reshape(db_sto, [1, n, nb_pcl]);

db_Mi_ss = Mi_ss + dt * db_deter - db_sto;

end

function db_Gr = evolve_Gr(pchol_cov_noises, Gr, tau, dt, nb_pcl, n)

% Evolve Gr with Euler-Maruyama
db_sto = repmat(pchol_cov_noises(1 : n, (n + 1) * n + 1 : end), [1, 1, nb_pcl]) ...
    .* randn(1, 1, nb_pcl) * sqrt(dt);
db_sto = reshape(db_sto, [1, n, n, nb_pcl]);
db_deter = -Gr / tau;

db_Gr = Gr + dt * db_deter + db_sto;

end


function noises = generate_noises(pchol_cov_noises, n, nb_pcl, dt)

noises = pchol_cov_noises * randn((n + 2) * n, nb_pcl) * sqrt(dt); % (n + 1) * n for theta_theta and n for xi_xi
noises = permute(noises, [1 3 4 2]); % (n + (n+1)*n) x 1 x 1 x nb_pcl

end
