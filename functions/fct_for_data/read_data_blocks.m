function param=read_data_blocks(type_data,folder_data)
% Get some information on how the data are saved
%

switch type_data
%     case 'DNS_2D_incompact3d_re1000'
%         param.data_in_blocks.bool=true; % data are saved in several files
%         param.data_in_blocks.nb_blocks=2; % data are saved in 5 files
%         param.data_in_blocks.len_blocks=920; % In each files, there are 100 snapshots
    case 'DNS100_inc3d_2D_2018_11_16_blocks'
        param.data_in_blocks.bool=true; % data are saved in several files
        param.data_in_blocks.nb_blocks=20; % data are saved in 10 files
        param.data_in_blocks.len_blocks=1000; % In each files, there are 100 snapshots
    case 'DNS100_inc3d_2D_2018_11_16_blocks_truncated'
        param.data_in_blocks.bool=true; % data are saved in several files
        param.data_in_blocks.nb_blocks=14; % data are saved in 14 files
        param.data_in_blocks.len_blocks=1000; % In each files, there are 1000 snapshots
    case 'DNS100_inc3d_2D_2018_11_16_blocks_test_basis'
        param.data_in_blocks.bool=true; % data are saved in several files
        param.data_in_blocks.nb_blocks=2; % data are saved in 2 files : 20 periods
%         param.data_in_blocks.nb_blocks=1; % data are saved in 1 files :
%         10 periods
% %         param.data_in_blocks.nb_blocks=10; % data are saved in 10 files
        param.data_in_blocks.len_blocks=1000; % In each files, there are 1000 snapshots
    case 'DNS100_OpenFOAM_2D_2020_blocks'
        param.data_in_blocks.bool=true; % data are saved in several files
        param.data_in_blocks.nb_blocks=17; % data are saved in 17 files : ~100 periods
        param.data_in_blocks.len_blocks=681; % In each files, there are 681 snapshots
    case 'DNS100_OpenFOAM_2D_2020_blocks_truncated'
        param.data_in_blocks.bool=true; % data are saved in several files
        param.data_in_blocks.nb_blocks=12; % data are saved in 17 files : ~100 periods
        param.data_in_blocks.len_blocks=681; % In each files, there are 681 snapshots
    case 'DNS100_OpenFOAM_2D_2020_blocks_test_basis'
        param.data_in_blocks.bool=true; % data are saved in several files
        param.data_in_blocks.nb_blocks=2; % data are saved in 17 files : ~100 periods
        param.data_in_blocks.len_blocks=681; % In each files, there are 681 snapshots
    case 'inc3D_Re300_40dt_blocks'
        param.data_in_blocks.bool=true; % data are saved in several files
        param.data_in_blocks.nb_blocks=5; % data are saved in 5 files
        param.data_in_blocks.len_blocks=100; % In each files, there are 100 snapshots
    case 'inc3D_Re300_40dt_blocks_truncated'
        param.data_in_blocks.bool=true; % data are saved in several files
        param.data_in_blocks.nb_blocks=4; % data are saved in 4 files
        param.data_in_blocks.len_blocks=100; % In each files, there are 100 snapshots
    case 'inc3D_Re300_40dt_blocks_test_basis'
        param.data_in_blocks.bool=true; % data are saved in several files
        param.data_in_blocks.nb_blocks=1; % data are saved in 1 files
        param.data_in_blocks.len_blocks=100; % In each files, there are 100 snapshots
    case 'small_test_in_blocks'
        param.data_in_blocks.bool=true; % data are saved in several files
        param.data_in_blocks.nb_blocks=5; % data are saved in 5 files
        param.data_in_blocks.len_blocks=2; % In each files, there are 100 snapshots
    case 'small_test_in_blocks_truncated'
        param.data_in_blocks.bool=true; % data are saved in several files
        param.data_in_blocks.nb_blocks=4; % data are saved in 4 files
        param.data_in_blocks.len_blocks=2; % In each files, there are 100 snapshots
    case 'small_test_in_blocks_test_basis'
        param.data_in_blocks.bool=true; % data are saved in several files
        param.data_in_blocks.nb_blocks=1; % data are saved in 1 files
        param.data_in_blocks.len_blocks=2; % In each files, there are 100 snapshots
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
        param.data_in_blocks.nb_blocks = 31; % data are saved in 31 files -> 31 vortex shedings
        param.data_in_blocks.len_blocks = 5; % In each files, there are 20 snapshots
    case 'inc3D_HRLESlong_Re3900_blocks_truncated'
        param.data_in_blocks.bool=true; % data are saved in several files
        param.data_in_blocks.nb_blocks = 28; 
        param.data_in_blocks.len_blocks = 5; % In each files, there are 20 snapshots
    case 'inc3D_HRLESlong_Re3900_blocks_test_basis'
        param.data_in_blocks.bool=true; % data are saved in several files
        param.data_in_blocks.nb_blocks = 3; 
        param.data_in_blocks.len_blocks = 5; % In each files, there are 20 snapshots
        
%     case 'inc3D_HRLESlong_Re3900_blocks'
%         param.data_in_blocks.bool=true; % data are saved in several files
%         param.data_in_blocks.nb_blocks=73; % data are saved in 73 files -> 73 vortex shedings
%         param.data_in_blocks.len_blocks=20; % In each files, there are 20 snapshots
%     case 'inc3D_HRLESlong_Re3900_blocks_truncated'
%         param.data_in_blocks.bool=true; % data are saved in several files
%         param.data_in_blocks.nb_blocks=65; % data are saved in 65 files -> 65 vortex shedings
%         param.data_in_blocks.len_blocks=20; % In each files, there are 20 snapshots
%     case 'inc3D_HRLESlong_Re3900_blocks_test_basis'
%         param.data_in_blocks.bool=true; % data are saved in several files
%         param.data_in_blocks.nb_blocks=8; % data are saved in 8 files -> 8 vortex shedings
%         param.data_in_blocks.len_blocks=20; % In each files, there are 20 snapshots
        
    case 'DNS300_inc3d_3D_2017_04_02_blocks'
        param.data_in_blocks.bool=true; % data are saved in several files
        param.data_in_blocks.nb_blocks=10; % data are saved in 10 files -> 80 vortex shedings
        param.data_in_blocks.len_blocks=160; % In each files, there are 160 snapshots -> 8 vortex shedings
    case 'DNS300_inc3d_3D_2017_04_02_blocks_truncated'
        param.data_in_blocks.bool=true; % data are saved in several files
        param.data_in_blocks.nb_blocks=8; % data are saved in 8 files -> 80 vortex shedings
        param.data_in_blocks.len_blocks=160; % In each files, there are 160 snapshots -> 8 vortex shedings
    case 'DNS300_inc3d_3D_2017_04_02_blocks_test_basis'
        param.data_in_blocks.bool=true; % data are saved in several files
        param.data_in_blocks.nb_blocks=2; % data are saved in 2 files -> 80 vortex shedings
        param.data_in_blocks.len_blocks=160; % In each files, there are 160 snapshots -> 8 vortex shedings
    case 'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks'
        param.data_in_blocks.bool=true; % data are saved in several files
        param.data_in_blocks.nb_blocks=100; % data are saved in 100 files -> 80 vortex shedings
        param.data_in_blocks.len_blocks=16; % In each files, there are 16 snapshots -> 1 vortex shedings
    case 'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_truncated'
        param.data_in_blocks.bool=true; % data are saved in several files
        param.data_in_blocks.nb_blocks=80; % data are saved in 80 files -> 80 vortex shedings
        param.data_in_blocks.len_blocks=16; % In each files, there are 16 snapshots -> 1 vortex shedings
    case 'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_test_basis'
        param.data_in_blocks.bool=true; % data are saved in several files
        param.data_in_blocks.nb_blocks=20; % data are saved in 20 files -> 80 vortex shedings
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
    case 'turb2D_blocks'
        param.data_in_blocks.bool=true; % data are saved in several files
        param.data_in_blocks.nb_blocks=14; % data are saved in 14 files -> ? vortex shedings
        param.data_in_blocks.len_blocks=1000; % In each files, there are 1000 snapshots -> ? vortex shedings
    case 'turb2D_blocks_truncated'
        param.data_in_blocks.bool=true; % data are saved in several files
        param.data_in_blocks.nb_blocks=10; % data are saved in 10 files -> ? vortex shedings
        param.data_in_blocks.len_blocks=1000; % In each files, there are 1000 snapshots -> ? vortex shedings
    case 'turb2D_blocks_test_basis'
        param.data_in_blocks.bool=true; % data are saved in several files
        param.data_in_blocks.nb_blocks=1; % data are saved in 4 files -> ? vortex shedings
        param.data_in_blocks.len_blocks=1000; % In each files, there are 1000 snapshots -> ? vortex shedings
    case 'test2D_blocks'
        param.data_in_blocks.bool=true; % data are saved in several files
        param.data_in_blocks.nb_blocks=4; % data are saved in 4 files -> ? vortex shedings
        param.data_in_blocks.len_blocks=10; % In each files, there are 10 snapshots -> ? vortex shedings
    case 'test2D_blocks_truncated'
        param.data_in_blocks.bool=true; % data are saved in several files
        param.data_in_blocks.nb_blocks=3; % data are saved in 3 files -> ? vortex shedings
        param.data_in_blocks.len_blocks=10; % In each files, there are 10 snapshots -> ? vortex shedings
    case 'test2D_blocks_test_basis'
        param.data_in_blocks.bool=true; % data are saved in several files
        param.data_in_blocks.nb_blocks=1; % data are saved in 1 files -> ? vortex shedings
        param.data_in_blocks.len_blocks=10; % In each files, there are 10 snapshots -> ? vortex shedings
end

param.type_data=type_data;
param.folder_data=folder_data;

end