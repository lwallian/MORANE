function name_file_ILC_deter=fct_name_file_ILC_deter(param,nb_modes)

name_file_ILC_deter=[ param.folder_data 'ILC_deter_' ...
    param.type_data '_' num2str(nb_modes) '_modes'];
if param.eq_proj_div_free > 0
    name_file_ILC_deter=[ name_file_ILC_deter '_DFSP'];    
end
name_file_ILC_deter=[ name_file_ILC_deter '.mat'];