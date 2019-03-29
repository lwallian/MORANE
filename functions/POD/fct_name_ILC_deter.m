function param = fct_name_ILC_deter(param)
% Create the name of the file where the 1st reuslts (the ROM coefficeints 
% associated with the deterministic NAvier-Stokes) are saved
%

param.name_file_ILC_deter=[ param.folder_results 'ILC_deter_' ...
    param.type_data '_' num2str(param.nb_modes) '_modes.mat'];
        