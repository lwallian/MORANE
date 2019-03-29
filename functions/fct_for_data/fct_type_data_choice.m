function type_data = fct_type_data_choice(igrida)
% Type of data used
%

if ~igrida % Small data : it can be used on a local computer or 
           % on the computing grid IGRIDA
    dbstop if error % Help debugging
    
    % Do not use these data:
    
% %     % type_data = 'test';
% %     % type_data = 'test2';
% %     % type_data = 'test3';
% %     % type_data = 'dominique_config13';
% %     % type_data = 'dominique_config12';
% %     % type_data = 'dominique_config12bis';
% 
% %          type_data = 'adda';
%          
% %     % type_data = 'incompact3D_noisy2D';
% %     % type_data = 'incompact3D_noisy2D_subsampl';
% %     % type_data = 'incompact3D_noisy2D_10dt';
% %     %     type_data = 'incompact3D_noisy2D_10dt_subsampl';
% %     % type_data = 'incompact3D_noisy2D_20dt_subsampl';
% % %     type_data = 'inc3D_Re300_40dt_blocks';
% %     
% %     % type_data = 'incompact3D_noisy2D_10dt_tronc';
% %     % type_data = 'incompact3D_noisy2D_10dt_subsampl_v7_3';
% %     % type_data = 'data_SQG_manu';
% %     % type_data = 'data_geostrophic_manu';
% 
%         type_data = 'incompact3d_wake_episode3_cut';
%         
% %     %     type_data = 'LES_3D_tot_sub_sample_blurred';
% %     %     type_data = 'LES_3D_1_sub_sample';
% %     %     type_data = 'test_3D2';
%     
% 
% %     % Small size data (2D) : to use in order to test the algorithm
%     type_data = 'incompact3D_noisy2D_40dt_subsampl'; 
%     
%     Same data but saved in 5 differnts files :
%     to use in order to test the algorithm
    type_data = 'inc3D_Re300_40dt_blocks';
% 
% %         type_data = 'DNS_2D_incompact3d_re1000';


%     % These 3D data are bigger, since the spatial grid is thinner
%     % and the number of time step is bigger
%     % (~ 2000 time step)
%     % They are saved in different files
%     % The internship should try to use these data
%    type_data = 'inc3D_Re3900_blocks_truncated';
%    warning('DNS3900 used with big_data=false');
   
   
%    type_data = 'incompact3d_wake_episode3_cut_truncated';
   
%    type_data = 'inc3D_HRLESlong_Re3900_blocks';
    
else % Big data : it should be used only on the computing grid IGRIDA
    
    % Do not use these data:
    
    % type_data = 'incompact3D_noisy2D_subsampl_repmat3D';
    % type_data = 'test_3D';
    % type_data = 'test_3D2';
    % type_data = 'LES_3D_1';
    % type_data = 'LES_3D_2';
    % type_data = 'LES_3D_3';
    % type_data = 'LES_3D_4';
    % type_data = 'LES_3D_1_sub_sample';
    %     type_data = 'LES_3D_tot_sub_sample';
    
    % These 3D data give good results
    % They are saved in only one file 
    % (~ 250 time step)
%      type_data = 'LES_3D_tot_sub_sample_blurred';

%     % These 3D data are bigger, since the spatial grid is thinner
%     % and the number of time step is bigger
%     % (~ 2000 time step)
%     % They are saved in different files
%     % The internship should try to use these data
%    type_data = 'inc3D_Re3900_blocks';
   
   type_data = 'inc3D_HRLESlong_Re3900_blocks';
   
   
end

type_data
