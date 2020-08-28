function globalTest()
% script to read pointField from C++ simulation Re 100 and convert them on
% the REDLUM code format

param.N_tot = 11577;
param.MX = [362 218];
LX = [ 20 12 ];
param.dX = [ LX(1)/(param.MX(1) - 1) LX(2)/(param.MX(2) - 1) ];
param.M = param.MX(1)*param.MX(2);
param.d = 2;

nb_file = int32(17);

exportFields_globalTest(param, nb_file);



end

