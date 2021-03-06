function b_tp2 = RK4_NLMEV(bt,ILC,dt)
% Compute b(t+2)
% The sizes of the inputs should be :
% - I = [ I(t) I(t+1) I(t+2)] : 3 x m 
% - L = [ L(t) L(t+1) L(t+2)] : 3 x m x m
% - C : m x m x m
% - bt = b(t) : 1 x m 
% The result has the size : N x m

k1 = deriv_bt_NLMEV( ILC, bt);
k2 = deriv_bt_NLMEV( ILC, bt + k1*dt/2);
k3 = deriv_bt_NLMEV( ILC, bt + k2*dt/2);
k4 = deriv_bt_NLMEV( ILC, bt + k3*dt);

b_tp2 = bt + (dt/3)*(k1/2 + k2 + k3 + k4/2);