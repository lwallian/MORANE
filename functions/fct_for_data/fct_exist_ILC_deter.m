function [bool, param, ILC_deter]= fct_exist_ILC_deter(param)
% Test if the ROM coefficients associated with the determinisitc
% Navier-Stokes are already computed

tic
MAX_possible_nb_modes = 100;
bool = false;
m = param.nb_modes;
k = m;
global stochastic_integration;
while (~ bool) && ( k <= MAX_possible_nb_modes )
%     name_file_ILC_deter=[ param.folder_results 'ILC_deter_' ...
    name_file_ILC_deter=[ param.folder_data 'ILC_deter_' ...
        param.type_data '_' num2str(k) '_modes_integ_' stochastic_integration '.mat'];
    bool = ( exist(name_file_ILC_deter,'file')==2 );
    k = k + 1;
end
% toc
if bool
    k = k -1;
    if k >= m
        load(name_file_ILC_deter,'L_deter','C_deter','C_deter_residu');
        
        L_deter = L_deter(1:m,1:m);
        C_deter = C_deter(1:m,1:m,1:m);
        C_deter_residu = C_deter_residu(1:m,1:m);
        param.C_deter_residu = C_deter_residu;
        
        I_deter=zeros(m,1);
        for i=1:m
            I_deter = I_deter - ...
                param.lambda(i)*squeeze(C_deter(i,i,:));
        end
        
        ILC_deter{1} = I_deter;
        ILC_deter{2} = L_deter;
        ILC_deter{3} = C_deter;
    end
else
    ILC_deter = nan;
end
