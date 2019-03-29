function param = fct_name_ref_Q(...
    param,reconstruction)
% Create the name of the file where the reconstruction of U are saved
%

param.name_file_Reconstruction_Q=[ param.folder_data 'Reconstruction_Q_' ...
    param.type_data '_' num2str(param.nb_modes) '_modes_'  ...
    'REF' ...
    '_decor_by_subsampl_' num2str(param.decor_by_subsampl.meth) ...
    '_choice_' num2str(param.decor_by_subsampl.choice_n_subsample)  ...
    '_threshold_' num2str(param.decor_by_subsampl.spectrum_threshold) ...
    'fct_test_' param.decor_by_subsampl.test_fct ];
if param.decor_by_subsampl.no_subampl_in_forecast
    param.name_file_Reconstruction_Q=[param.name_file_Reconstruction_Q '_no_subampl_in_forecast'];
end
if reconstruction
    param.reconstruction=true;
    param.name_file_Reconstruction_Q=[param.name_file_Reconstruction_Q '_reconstruction'];
else
    param.reconstruction=false;
end
param.name_file_Reconstruction_Q = [ param.name_file_Reconstruction_Q '/' ];
mkdir(param.name_file_Reconstruction_Q);


%%

% param.name_file_Reconstruction_Q=[ param.folder_data 'Reconstruction_Q_' ...
%         param.type_data '_' num2str(param.nb_modes) '_modes_' ...
%         'REF' ];
% if reconstruction
%     param.reconstruction=true;
%     param.name_file_Reconstruction_Q=[param.name_file_Reconstruction_Q '_reconstruction'];
% else
%     param.reconstruction=false;
% end
% param.name_file_Reconstruction_Q = [ param.name_file_Reconstruction_Q '/'];
% mkdir(param.name_file_Reconstruction_Q);