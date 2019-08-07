function [param] = mock_param_structure(type_data, is_in_blocks)
%MOCK_PARAM_STRUCTURE Generates a parameter structure depending on the data
%type
%   @param type_data: dataset to test on
%   @param is_in_blocks: the dataset in distributed in different files
%   @return param: the generated parameter structure
%
% Author: Agustin PICARD, intern @ Scalian with Valentin RESSEGUIER as
% supervisor
%

%% NOT FINISHED YET!!!
% use ideas from fct_def_param

clear param;
param.type_data = type_data;
param.folder_results = [ pwd '/resultats/current_results/'];
param.folder_data = [ pwd '/data/'];

% Calculate N_tot
if is_in_blocks
    param_blocks = read_data_blocks(param_ref.type_data, param_ref.folder_data);
    nb_blocks = param_blocks.data_in_blocks.nb_blocks;
    len_blocks = param_blocks.data_in_blocks.len_blocks;
    param.N_tot = len_blocks * nb_blocks;
else
    name_file_U_centered=[param_ref.folder_data param_ref.type_data '_U_centered'];
    load(name_file_U_centered, 'U');
    [param.M , param.N_tot, param.d] = size(U);
    clear U;
end

end

