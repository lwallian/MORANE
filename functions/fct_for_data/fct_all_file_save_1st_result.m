function [bool,I_deter,L_deter,C_deter,param_ref] ...
    = fct_all_file_save_1st_result(param_ref)
% Create the file name to save the first results of the POD
%
a_t = [true false];

v_threshold=[0.0005 [10 1 0.1]/1000];
%     v_threshold=[0.1 1 10]/1000;

% warning('These choices of threeshoold are specific to a type of data');

%    v_threshold=0.0005 % for inc DNS 300 episode3


files_save = [ ];
for p=1:2
    for q=1:length(v_threshold)
        param_temp = param_ref;
        param_temp.a_time_dependant=a_t(p);
        param_temp.decor_by_subsampl.spectrum_threshold = v_threshold(q);
        name_file_temp = fct_file_save_1st_result(param_temp);
        files_save = [ files_save {name_file_temp} ];
    end
end

%%

bool = false;
k=1;
while (~ bool) && ( k <= length(files_save) )
    name_file_temp = files_save{k};
    bool = ( exist(name_file_temp,'file')==2 );
    k = k + 1;
end
% toc
if bool
    load(name_file_temp,'I_deter','L_deter','C_deter','param');
    param_ref.C_deter_residu = param.C_deter_residu;
    clear param
else
    I_deter=nan;
    L_deter=nan;
    C_deter=nan;
end
