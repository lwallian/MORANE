function bt_evol = evol_forward_bt_SSPRK3_3(I, L, C, dt, bt)
% Compute the next bt through SSP RK (3,3) integration
% The sizes of the inputs should be :
% - I : 2N-1 x m
% - L : 2N-1 x m x m
% - C : m x m x m
% - bt : N x m 
% The result has the size : 1 x m
%
% Author: Agustin PICARD, intern @ Scalian with Valentin RESSEGUIER as
% supervisor
%

% Modify sizes
N = size(bt,1);
if iscolumn(I)
    I = repmat(permute(I,[2 1]),[max(2*N-1,3) 1]);
else
    I=I';
end
if ismatrix(L)
    L = repmat(permute(L,[3 1 2]),[max(2*N-1,3) 1 1]);
else
    L=permute(L,[3 1 2]);
end

% Time integration by 4-th order Runge Kutta 
if N<1
    error('bt is empty');
else
    bt_evol = SSPRK3_3(bt(end,:),I(end-2:end,:),L(end-2:end,:,:),C,dt);
end

end

function b_tp1 = SSPRK3_3(bt,I,L,C,dt)

k1 = deriv_bt( I(1,:)', squeeze(L(1,:,:)), C, bt);
k2 = deriv_bt( I(2,:)', squeeze(L(2,:,:)), C, bt + k1*dt/2);
k3 = deriv_bt( I(3,:)', squeeze(L(3,:,:)), C, bt + k2*dt);

u1 = bt + dt * k1;
u2 = 3 / 4 * bt + u1 / 4 + dt * k2 / 4;

b_tp1 = (bt / 3) + (2 / 3) * (u2 + dt * k3);

end

