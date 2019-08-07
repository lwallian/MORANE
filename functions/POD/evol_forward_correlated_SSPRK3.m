function [bt_evol, eta, Mi_ss, Gr] = evol_forward_correlated_SSPRK3(I, L, C, ...
                        pchol_cov_noises, tau, dt, bt, eta, Gr, Mi_ss)
%EVOL_FORWARD_CORRELATED_SSPRK3 Evolves the correlated velocity model using
%an SSPRK3 integration scheme
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
[eta, Mi_ss, Gr] = evolve_sto_coeff(noises, Mi_sigma, tau, eta, Mi_ss, Gr, n);

% Do the SSPRK3 integration by steps
db_m = evolve_sto(bt, Mi_ss, eta);
db_m = permute( db_m , [2 1 4 3]);
k1 = evolve_deter(bt, I, L ,C);
k1 = permute( k1 , [2 1 4 3]);
u1 = bt + dt * k1 + db_m;
k2 = evolve_deter(u1, I, L ,C);
k2 = permute( k2 , [2 1 4 3]);
db_m = evolve_sto_coeff(noises, Mi_sigma, n, nb_pcl, bt);
db_m = permute( db_m , [2 1 4 3]);
u2 = 3 / 4 * bt + u1 / 4 + dt * k2 / 4 + db_m / 4;
k3 = evolve_deter(u2, I, L ,C);
k3 = permute( k3 , [2 1 4 3]);
db_m = evolve_sto_coeff(noises, Mi_sigma, n, nb_pcl, bt);
db_m = permute( db_m , [2 1 4 3]);

bt_evol = (bt / 3) + (2 / 3) * (u2 + dt * k3 + db_m);
bt_evol = permute( bt_evol , [2 1 4 3]);

end


function db_fv = evolve_deter(bt, I, L, C)

bt = permute(bt,[2 1 4 3]); % m x 1 x 1 x nb_pcl

C = bsxfun(@times,bt,C); % m x m x m x nb_pcl
C = permute(sum(C,1),[2 3 1 4]); % m x m x 1 x nb_pcl


C = bsxfun(@times,bt,C); % m x m x 1 x nb_pcl
C = permute(sum(C,1),[2 3 1 4]); % m x 1 x 1 x nb_pcl

L = bsxfun(@times,bt,L); % m x m x 1 x nb_pcl
L = permute(sum(L,1),[2 3 1 4]); % m x 1 x 1 x nb_pcl

db_fv = - bsxfun(@plus, I, L + C ) ; % m x 1 x 1

end


function db_m = evolve_sto(bt, Mi_ss, eta)

db_m = eta * bt + Mi_ss; % permutes maybe

end


function [db_eta, db_Mi_ss, db_Gr] = evolve_sto_coeff(noises, Mi_sigma, tau, eta, Mi_ss, Gr, n)

% Evolve both eta_i and M_i_ss
db_eta = evolve_eta(noises, tau, eta, n, dt);
[db_Mi_ss, db_Gr] = evolve_Mi_ss(noises, tau, Mi_sigma, Mi_ss, Gr, n, dt);

end


function db_eta = evolve_eta(noises, tau, eta, n, dt)

% Evolve eta with SSPRK3
db_sto = noises(n + 1 : end, 1, 1, :); % permute maybe ?
db_deter = -eta / tau;
u1 = eta + dt * db_deter + db_sto;
db_deter = -u1 / tau;
u2 = 3 / 4 * eta + u1 / 4 + dt * db_deter / 4 + db_sto / 4;
db_deter = -u2 / tau;

db_eta = eta / 3 + (2 / 3) * (u2 + dt * db_deter + db_sto);

end


function [db_Mi_ss, db_Gr] = evolve_Mi_ss(noises, tau, Mi_sigma, Mi_ss, Gr, n, dt)

% Evolve Mi_ss with SSPRK3
mi_ss_noise = randn(n, 1) * sqrt(dt);
db_Gr = evolve_Gr(noises, Gr, tau, dt);
db_deter = -Mi_ss / tau + Mi_sigma;
db_sto = db_Gr * mi_ss_noise;
u1 = Mi_ss + dt * db_deter + db_sto;
db_deter = -u1 / tau + Mi_sigma;
u2 = 3 / 4 * Mi_ss + u1 / 4 + dt * db_deter / 4 + db_sto / 4;
db_deter = -u2 / tau + Mi_sigma;

db_Mi_ss = Mi_ss / 3 + (2 / 3) * (u2 + dt * db_deter + db_sto);

end


function db_Gr = evolve_Gr(noises, Gr, tau, dt)

% Evolve Gr with SSPRK3
db_sto = noises(1 : n, 1 , 1, :);
db_deter = -Gr / tau;
u1 = Gr + dt * db_deter + db_sto;
db_deter = -u1 / tau;
u2 = 3 / 4 * Gr + u1 / 4 + dt * db_deter / 4 + db_sto / 4;
db_deter = -u2 / tau;

db_Gr = Gr / 3 + (2 / 3) * (u2 + dt * db_deter + db_sto);

end


function noises = generate_noises(pchol_cov_noises, n, nb_pcl, dt)

noises = pchol_cov_noises*randn((n+1)*n,nb_pcl)*sqrt(dt);
noises = permute(noises,[1 3 4 2]); % (n+1)*n x nb_pcl
clear pchol_cov_noises; % (n+1)*n x 1 x 1 x nb_pcl

end
