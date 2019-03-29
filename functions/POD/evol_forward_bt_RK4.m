function bt_evol = evol_forward_bt_RK4(I,L,C, dt, bt)
% Compute the next bt
% The sizes of the inputs should be :
% - I : 2N-1 x m
% - L : 2N-1 x m x m
% - C : m x m x m
% - bt : N x m 
% The result has the size : 1 x m
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
    bt_evol = RK4(bt(end,:),I(end-2:end,:),L(end-2:end,:,:),C,dt);
end