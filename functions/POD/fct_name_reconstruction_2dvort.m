function param = fct_name_reconstruction_2dvort(...
    param,modal_dt,reconstruction,name_simu)
% Create the name of the file where the reconstruction of U are saved
%

if strcmp(name_simu,'ref')
    param.name_file_Reconstruction_omega=[ param.folder_data 'Reconstruction_2dvort_' ...
        param.type_data '_' num2str(param.nb_modes) '_modes_'  ...
        'REF' ...
        '_decor_by_subsampl_' num2str(param.decor_by_subsampl.meth) ...
        '_choice_' num2str(param.decor_by_subsampl.choice_n_subsample)  ...
        '_threshold_' num2str(param.decor_by_subsampl.spectrum_threshold) ...
        'fct_test_' param.decor_by_subsampl.test_fct ];
    if param.decor_by_subsampl.no_subampl_in_forecast
        param.name_file_Reconstruction_omega=[param.name_file_Reconstruction_omega '_no_subampl_in_forecast'];
    end
    if reconstruction
        param.reconstruction=true;
        param.name_file_Reconstruction_omega=[param.name_file_Reconstruction_omega '_reconstruction'];
    else
        param.reconstruction=false;
    end
    param.name_file_Reconstruction_omega = [ param.name_file_Reconstruction_omega '/' ];
else
    if param.a_time_dependant
        dependance_on_time_of_a = '_a_time_dependant_';
    else
        dependance_on_time_of_a = '_a_cst_';
    end
    if param.decor_by_subsampl.bool
        if strcmp(dependance_on_time_of_a,'a_t')
            char_filter = [ '_on_' param.type_filter_a ];
        else
            char_filter = [];
        end
        param.name_file_Reconstruction_omega=[ param.folder_data 'Reconstruction_2dvort_' ...
            param.type_data '_' num2str(param.nb_modes) '_modes_' ...
            dependance_on_time_of_a char_filter ...
            '_decor_by_subsampl_' num2str(param.decor_by_subsampl.meth) ...
            '_choice_' num2str(param.decor_by_subsampl.choice_n_subsample)  ...
            '_threshold_' num2str(param.decor_by_subsampl.spectrum_threshold) ...
            'fct_test_' param.decor_by_subsampl.test_fct ];
    else
        param.name_file_Reconstruction_Q=[ param.folder_data 'Reconstruction_2dvort_' ...
            param.type_data '_' num2str(param.nb_modes) '_modes_' ...
            dependance_on_time_of_a ];
    end
    param.name_file_Reconstruction_omega=[param.name_file_Reconstruction_omega '_fullsto'];
    % if modal_dt
    %     param.name_file_Reconstruction_omega=[param.name_file_Reconstruction_omega '_modal_dt'];
    % end
    if modal_dt == 1
        param.name_file_Reconstruction_omega=[param.name_file_Reconstruction_omega '_modal_dt'];
    elseif modal_dt == 2
        param.name_file_Reconstruction_omega=[param.name_file_Reconstruction_omega '_real_dt'];
    end
    if ~ param.adv_corrected
        param.name_file_Reconstruction_omega=[param.name_file_Reconstruction_omega '_no_correct_drift'];
    end
    if param.decor_by_subsampl.no_subampl_in_forecast
        param.name_file_Reconstruction_omega=[param.name_file_Reconstruction_omega '_no_subampl_in_forecast'];
    end
    if reconstruction
        param.reconstruction=true;
        param.name_file_Reconstruction_omega=[param.name_file_Reconstruction_omega '_reconstruction'];
    else
        param.reconstruction=false;
    end
    param.name_file_Reconstruction_omega = [ param.name_file_Reconstruction_omega '/' ...
        name_simu '/'];
end
mkdir(param.name_file_Reconstruction_omega);

