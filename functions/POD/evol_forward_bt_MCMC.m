function [bt_evol,bt_fv,bt_m] = evol_forward_bt_MCMC(I,L,C, ...
                        pchol_cov_noises, dt, bt,bt_fv,bt_m)
% Compute the next bt
% The sizes of the inputs should be :
% - I : m
% - L : m x m
% - C : m x m x m
% - bt : 1 x m x nb_pcl
% The result has the size : 1 x m
%

[~ , n , nb_pcl ] = size(bt);

bt = permute(bt,[2 1 4 3]); % m x 1 x 1 x nb_pcl

C = bsxfun(@times,bt,C); % m x m x m x nb_pcl
C = permute(sum(C,1),[2 3 1 4]); % m x m x 1 x nb_pcl


C = bsxfun(@times,bt,C); % m x m x 1 x nb_pcl
C = permute(sum(C,1),[2 3 1 4]); % m x 1 x 1 x nb_pcl

L = bsxfun(@times,bt,L); % m x m x 1 x nb_pcl
L = permute(sum(L,1),[2 3 1 4]); % m x 1 x 1 x nb_pcl

d_b_fv = - bsxfun(@plus, I, L + C )*dt ; % m x 1 x 1
clear I L C

noises=pchol_cov_noises*randn((n+1)*n,nb_pcl)*sqrt(dt);
noises=permute(noises,[1 3 4 2]); % (n+1)*n x nb_pcl
clear pchol_cov_noises; % (n+1)*n x 1 x 1 x nb_pcl
theta_alpha0_dB_t = noises(1:n,1,1,:); % n(i) x 1 x 1 x nb_pcl
alpha_dB_t =reshape(noises(n+1:end,1,1,:),[n n 1 nb_pcl]); % n(j) x n(i) x 1 x nb_pcl


%%%%%%%%%%%%%%%%%%

% s = size(alpha_dB_t,4)
% alpha_dB_t(:,:,1,:) = zeros(8,8,s)
% alpha_dB_t(:,:,1,1) = ones(8,8)*2.2
% 
% theta_alpha0_dB_t(:,:,1,:) = zeros(8,1,1,s)
% theta_alpha0_dB_t(:,:,1,1) = ones(8,1)*0.05

%%%%%%%%%%%%%%%%%%


clear noises

alpha_dB_t = bsxfun(@times,bt,alpha_dB_t); % m(j) x m(i) x 1 x nb_pcl
alpha_dB_t = permute(sum(alpha_dB_t,1),[2 3 1 4]); % m(i) x 1 x 1 x nb_pcl

d_b_m = alpha_dB_t + theta_alpha0_dB_t;% m(i) x 1 x 1 x nb_pcl

if nargin >6
    bt_fv = bt_fv + permute(d_b_fv  , [2 1 4 3]);
    bt_m = bt_m + permute(d_b_m  , [2 1 4 3]);
end
bt_evol = bt + d_b_fv + d_b_m ;
bt_evol = permute( bt_evol , [2 1 4 3]);

































