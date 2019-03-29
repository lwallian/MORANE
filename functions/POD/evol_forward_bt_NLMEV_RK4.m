function bt_evol = evol_forward_bt_NLMEV_RK4(ILC, dt, bt)
% Compute the next bt
% The sizes of the inputs should be :
% - I : 2N-1 x m
% - L : 2N-1 x m x m
% - C : m x m x m
% - bt : N x m 
% The result has the size : 1 x m
%

bt_evol = RK4_NLMEV(bt(end,:),ILC,dt);

end