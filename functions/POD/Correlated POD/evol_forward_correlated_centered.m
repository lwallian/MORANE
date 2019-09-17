function [bt_evol, b_fv, b_m, eta, Mi_ss, spiral] = evol_forward_correlated_centered(I, L, C, ...
                        pchol_cov_noises, tau, dt, bt, eta, spiral, Mi_ss, bt_fv, bt_m)
%EVOL_FORWARD_CORRELATED_CENTERED Evolves the correlated centered chronos using
%an Euler-Maruyama integration scheme
%   @param I, L, C: deterministic model coefficients
%   @param pchol_cov_noises: noise coefficients for the stochastic part
%   @param tau: decay rate for the Ornstein-Uhlenbeck equations
%   @param dt: time step
%   @param bt: last chronos states
%   @param eta: last multiplicative noise coefficients
%   @param spiral: last noise evolution coefficient for Mi_ss
%   @return bt_evol: current chronos states
%   @return eta: current multiplicative noise coefficients
%   @return Mi_ss: current additive noise coefficients
%   @return spiral: current noise evolution coefficient for Mi_ss

[~ , n , nb_pcl ] = size(bt);
noises = generate_noises(pchol_cov_noises, n, nb_pcl, dt);
[eta, Mi_ss, spiral] = evolve_sto_coeff(noises, ...
    tau, eta, Mi_ss, spiral, n, nb_pcl, dt);
clear noises;

% Evolve the equation with Euler-Maruyama
db_m = evolve_sto(bt, Mi_ss, eta, n, nb_pcl, dt);
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

bt = permute(bt,[2 1 4 3]); % m x 1 x 1 x nb_pcl

C = bsxfun(@times,bt,C); % m x m x m x nb_pcl
C = permute(sum(C,1),[2 3 1 4]); % m x m x 1 x nb_pcl


C = bsxfun(@times,bt,C); % m x m x 1 x nb_pcl
C = permute(sum(C,1),[2 3 1 4]); % m x 1 x 1 x nb_pcl

L = bsxfun(@times,bt,L); % m x m x 1 x nb_pcl
L = permute(sum(L,1),[2 3 1 4]); % m x 1 x 1 x nb_pcl

db_fv = - bsxfun(@plus, I, L + C ) * dt ; % m x 1 x 1

end


function db_m = evolve_sto(bt, Mi_ss, eta, n, nb_pcl, dt)

bt_x = permute(cat(2, bt, ones(1, 1, nb_pcl)), [1 2 4 3]);
eta_bt = bsxfun(@times, eta, bt_x);
eta_bt = sum(eta_bt, 2);
db_m = dt * reshape(eta_bt, [1, n, nb_pcl]) + dt * Mi_ss;

end



function [d_eta, d_Mi_ss, d_Gr] = evolve_sto_coeff(noises, tau, eta, Mi_ss, spiral, n, nb_pcl, dt)


% Evolve both eta_i and M_i_ss
d_eta = evolve_eta(noises, tau, eta, n, nb_pcl, dt);
[d_Mi_ss, d_Gr] = evolve_Mi_ss(noises, tau, Mi_ss, spiral, n, nb_pcl, dt);

end


function d_eta = evolve_eta(noises, tau, eta, n, nb_pcl, dt)

% Evolve eta with Euler-Maruyama
sto = reshape(noises(n + 1 : end, 1, 1, :), [1, n + 1, n, nb_pcl]);
deter = -eta / tau;

d_eta = eta + dt * deter + sto;

end


function [d_Mi_ss, db_spiral] = evolve_Mi_ss(noises, tau, Mi_ss, spiral, n, nb_pcl, dt)

% Evolve Mi_ss with Euler-Maruyama
mi_ss_noise = noises(1 : n, :, :, :);
% mi_ss_noise = noises(1 : n, :);
db_spiral = evolve_spiral(spiral, tau, dt, nb_pcl);
deter = - 2 * Mi_ss / tau;
sto = bsxfun(@times, db_spiral, mi_ss_noise);
sto = sum(sto, 3);
sto = reshape(sto, [1, n, nb_pcl]);

d_Mi_ss = Mi_ss + dt * deter + sto;

end

function d_spiral = evolve_spiral(spiral, tau, dt, nb_pcl)

% Evolve spiral with Euler-Maruyama
sto = sqrt(dt) * sqrt(2 / tau) * randn(1, 1, nb_pcl);
deter = -spiral / tau;

d_spiral = spiral + dt * deter + sto;

end


function noises = generate_noises(pchol_cov_noises, n, nb_pcl, dt)

noises = pchol_cov_noises * randn((n + 2) * n, nb_pcl) * sqrt(dt); % (n + 1) * n for theta_theta and n for xi_xi
noises = permute(noises, [1 3 4 2]); % (n + (n+1)*n) x 1 x 1 x nb_pcl

end

