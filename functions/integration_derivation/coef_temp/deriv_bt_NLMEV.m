function d_b = deriv_bt_NLMEV(ILC, bt)
% Compute the time derivation of bt
% The sizes of the inputs should be :
% - I : m
% - L : m x m
% - C : m x m x m
% - bt : N x m 
% The result has the size : N x m
%

L_EV=ILC.L_EV;
I=ILC.I;
L=ILC.L;
C=ILC.C;
N=1;

bt=bt';

C = bsxfun(@times,bt,C); % m x m x m x N
C = squeeze(sum(C,1)); % m x m x N

C = bsxfun(@times,bt,C); % m x m x N
C = sum(C,1); % 1 x m x N
C = permute(C,[2 3 1]); % m x N

K_T = norm(bt);
L_temp = L + K_T*L_EV;
L=L_temp;

L = bsxfun(@times,bt,L); % m x m x N
L = sum(L,1); % 1 x m x N
L = permute(L,[2 3 1]); % m x N

d_b = - ( I + L + C )' ; % N x m
