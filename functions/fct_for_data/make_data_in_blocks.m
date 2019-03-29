function make_data_in_blocks()

folder_data = '/Users/vressegu/Documents/matlab/POD-NS_Stochastique/current_used/data/';
type_data = 'incompact3D_noisy2D_40dt_subsampl';

folder_data_in_blocks = 'data_test_in_blocks';

folder_data_ref=folder_data;
%% Load data

load([ folder_data 'data_incompact3d_wake_noisy_1_40dt_sub_sampl.mat']);
% 
% N_test = 499;
% 
% [Mx,My,N_tot,d]=size(U);
% MX=[Mx My];
% M=prod(MX);
% dx=dX(1);
% dy=dX(2);
% a = dx*(1:Mx)';
% b = dy*(1:My);
% grid = {a b'};
% U=reshape(U,[Mx*My N_tot d]);

U_tot=U;
clear U

nb_blocks = 5;
nb_by_block = 100;

for big_T=1:nb_blocks
%     ((big_T-1)*nb_by_block +1)
%     big_T*nb_by_block
    U = U_tot(:,:,((big_T-1)*nb_by_block +1):big_T*nb_by_block,:);
    save([ folder_data folder_data_in_blocks '/inc40Dt_' num2str(big_T) ],'U','dt','Re','dX','normalized','-v7.3');
end
    
clear U U_tot





