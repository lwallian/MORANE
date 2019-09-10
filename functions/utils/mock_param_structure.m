function [param] = mock_param_structure(type_data, is_in_blocks, nb_modes)
%MOCK_PARAM_STRUCTURE Generates a parameter structure depending on the data
%type
%   @param type_data: dataset to test on
%   @param is_in_blocks: the dataset in distributed in different files
%   @return param: the generated parameter structure
%
% Author: Agustin PICARD, intern @ Scalian with Valentin RESSEGUIER as
% supervisor
%

clear param;
param.type_data = type_data;
param.data_in_blocks.bool = is_in_blocks;
param.folder_results = [ pwd '/resultats/current_results/'];
param.folder_data = [ '../data/'];

% Calculate N_tot
if is_in_blocks
    param_blocks = read_data_blocks(type_data, folder_data);
    nb_blocks = param_blocks.data_in_blocks.nb_blocks;
    len_blocks = param_blocks.data_in_blocks.len_blocks;
    param.data_in_blocks.nb_blocks = nb_blocks;
    param.data_in_blocks.len_blocks = len_blocks;
    param.N_tot = len_blocks * nb_blocks;
    clear param_blocks;
else
    name_file_U_centered=[param.folder_data param.type_data '_U_centered'];
    load(name_file_U_centered, 'U');
    [param.M , param.N_tot, param.d] = size(U);
    param.big_data = true;
    clear U;
end

param.nb_modes = nb_modes;
param.coef_correctif_estim.learn_coef_a = false;
param.eq_proj_div_free = 1;

param.decor_by_subsampl.bool = true;
param.decor_by_subsampl.n_subsampl_decor=nan;
param.decor_by_subsampl.meth='bt_decor';
param.decor_by_subsampl.choice_n_subsample='auto_shanon';
param.decor_by_subsampl.spectrum_threshold=1e-4;
param.decor_by_subsampl.test_fct = 'b';
param.a_time_dependant = false;
param.save_all_bi = false;

param.name_file_mode=['mode_' param.type_data '_' num2str(param.nb_modes) '_modes.mat'];
param.name_file_mode=[ param.folder_data param.name_file_mode ];
param.adv_corrected = false;
param.save_bi_before_subsampling = false;

end

