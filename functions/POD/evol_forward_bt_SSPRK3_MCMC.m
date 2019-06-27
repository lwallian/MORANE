function [bt_evol] = evol_forward_bt_SSPRK3_MCMC(I,L,C, ...
                        pchol_cov_noises, dt, bt)
% Compute the next bt
% The sizes of the inputs should be :
% - I : m
% - L : m x m
% - C : m x m x m
% - bt : 1 x m x nb_pcl
% The result has the size : 1 x m
%
% Author: Agustin PICARD, intern @ Scalian with Valentin RESSEGUIER as
% supervisor
%

% Do the SSPRK3 integration by steps
k1 = evolve_noisy_bt(bt, I, L ,C, pchol_cov_noises, dt);
k2 = evolve_noisy_bt(bt + k1 * dt / 2, I, L ,C, pchol_cov_noises, dt);
k3 = evolve_noisy_bt(bt + k2*dt, I, L ,C, pchol_cov_noises, dt);

u1 = bt + dt * k1;
u2 = 3 / 4 * bt + u1 / 4 + dt * k2 / 4;

bt_evol = (bt / 3) + (2 / 3) * (u2 + dt * k3);

end

function db = evolve_noisy_bt(bt, I, L, C, pchol_cov_noises, dt)

[~ , n , nb_pcl ] = size(bt);

db_fv = evolve_deter(bt, I, L, C);
db_m = generate_noise(pchol_cov_noises, n, nb_pcl, dt, bt);

db = db_fv + db_m;
db = permute( db , [2 1 4 3]);

end

function db_m = generate_noise(pchol_cov_noises, n, nb_pcl, dt, bt)

bt = permute(bt,[2 1 4 3]); % m x 1 x 1 x nb_pcl

noises=pchol_cov_noises*randn((n+1)*n,nb_pcl)*sqrt(dt);
noises=permute(noises,[1 3 4 2]); % (n+1)*n x nb_pcl
clear pchol_cov_noises; % (n+1)*n x 1 x 1 x nb_pcl
theta_alpha0_dB_t = noises(1:n,1,1,:); % n(i) x 1 x 1 x nb_pcl
alpha_dB_t =reshape(noises(n+1:end,1,1,:),[n n 1 nb_pcl]); % n(j) x n(i) x 1 x nb_pcl
clear noises

alpha_dB_t = bsxfun(@times,bt,alpha_dB_t); % m(j) x m(i) x 1 x nb_pcl
alpha_dB_t = permute(sum(alpha_dB_t,1),[2 3 1 4]); % m(i) x 1 x 1 x nb_pcl

db_m = alpha_dB_t + theta_alpha0_dB_t;% m(i) x 1 x 1 x nb_pcl

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

