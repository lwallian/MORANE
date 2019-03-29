function param=read_data_blocks(type_data,folder_data)
% Get some information on how the data are saved
%

switch type_data
%     case 'DNS_2D_incompact3d_re1000'
%         param.data_in_blocks.bool=true; % data are saved in several files
%         param.data_in_blocks.nb_blocks=2; % data are saved in 5 files
%         param.data_in_blocks.len_blocks=920; % In each files, there are 100 snapshots
    case 'inc3D_Re300_40dt_blocks'
        param.data_in_blocks.bool=true; % data are saved in several files
        param.data_in_blocks.nb_blocks=5; % data are saved in 5 files
        param.data_in_blocks.len_blocks=100; % In each files, there are 100 snapshots
    case 'inc3D_Re3900_blocks'
        param.data_in_blocks.bool=true; % data are saved in several files
%         param.data_in_blocks.nb_blocks=117;
        param.data_in_blocks.nb_blocks=119; % data are saved in 119 files -> 40 vortex shedings
        param.data_in_blocks.len_blocks=21; % In each files, there are 21 snapshots
    case 'inc3D_Re3900_blocks_truncated'
        param.data_in_blocks.bool=true; % data are saved in several files
%         param.data_in_blocks.nb_blocks=117;
        param.data_in_blocks.nb_blocks=90; % data are saved in 119 files -> 30 vortex shedings
        param.data_in_blocks.len_blocks=21; % In each files, there are 21 snapshots
%     case 'inc3D_Re3900_blocks_test'
%         param.data_in_blocks.bool=true; % data are saved in several files
%         param.data_in_blocks.nb_blocks=15; % data are saved in 119 files -> 30 vortex shedings
%         param.data_in_blocks.len_blocks=21; % In each files, there are 21 snapshots
    case 'inc3D_Re3900_blocks_test_basis'
        param.data_in_blocks.bool=true; % data are saved in several files
%         param.data_in_blocks.nb_blocks=117;
        param.data_in_blocks.nb_blocks=29; % data are saved in 119 files -> 30 vortex shedings
        param.data_in_blocks.len_blocks=21; % In each files, there are 21 snapshots
%     case 'inc3D_Re3900_blocks_test'
%         param.data_in_blocks.bool=true; % data are saved in several files
%         param.data_in_blocks.nb_blocks=15; % data are saved in 119 files -> 30 vortex shedings
%         param.data_in_blocks.len_blocks=21; % In each files, there are 21 snapshots
    case 'inc3D_HRLESlong_Re3900_blocks'
        param.data_in_blocks.bool=true; % data are saved in several files
        param.data_in_blocks.nb_blocks=73; % data are saved in 73 files -> 73 vortex shedings
        param.data_in_blocks.len_blocks=20; % In each files, there are 20 snapshots
    case 'DNS300_inc3d_3D_2017_04_02_blocks'
        param.data_in_blocks.bool=true; % data are saved in several files
        param.data_in_blocks.nb_blocks=10; % data are saved in 10 files -> 80 vortex shedings
        param.data_in_blocks.len_blocks=160; % In each files, there are 160 snapshots -> 8 vortex shedings
    case 'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks'
        param.data_in_blocks.bool=true; % data are saved in several files
        param.data_in_blocks.nb_blocks=100; % data are saved in 100 files -> 80 vortex shedings
        param.data_in_blocks.len_blocks=16; % In each files, there are 16 snapshots -> 1 vortex shedings
    case 'DNS300_inc3d_3D_2017_04_09_NOT_BLURRED_blocks'
        param.data_in_blocks.bool=true; % data are saved in several files
        param.data_in_blocks.nb_blocks=60; % data are saved in 60 files -> ? vortex shedings
        param.data_in_blocks.len_blocks=20; % In each files, there are 20 snapshots -> 1 vortex shedings
    case 'DNS300_inc3d_3D_2017_04_09_blocks'
        param.data_in_blocks.bool=true; % data are saved in several files
        param.data_in_blocks.nb_blocks=6; % data are saved in 60 files -> ? vortex shedings
        param.data_in_blocks.len_blocks=200; % In each files, there are 20 snapshots -> 1 vortex shedings
    case 'DNS100_inc3d_2D_2017_04_29_blocks'
        param.data_in_blocks.bool=true; % data are saved in several files
        param.data_in_blocks.nb_blocks=10; % data are saved in 60 files -> ? vortex shedings
        param.data_in_blocks.len_blocks=1000; % In each files, there are 20 snapshots -> 1 vortex shedings
end

param.type_data=type_data;
param.folder_data=folder_data;

end