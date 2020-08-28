function test_variance_tensor()

param.d = 2;
param.dX = [0.0554017 0.0552995];
param.MX = [362 218];
param.M = 362*218;
param.nb_modes = 2;
param.N_tot = 1052;
param.a_time_dependant = false;
param.dt = 0.05;
param.decor_by_subsampl.choice_n_subsample = 'htgen2';
param.folder_data = [ pwd '/data/'];
param.type_data = 'test_variance_tensor';
param.big_data = true;
param.data_in_blocks.bool = true;
param.data_in_blocks.len_blocks = 1;
param.decor_by_subsampl.n_subsampl_decor = 11;
param.lambda = 1;
param.decor_by_subsampl.meth = 'bt_decor';
param.decor_by_subsampl.spectrum_threshold = 1;
param.decor_by_subsampl.test_fct = 'b';
param.decor_by_subsampl.bool = false;
%name_file_U_temp = [param.folder_file_U_temp 'dsamp_' num2str(param.n_subsampl_decor) '_' ...
 %           num2str(big_T) '_U_temp'];
  %      param_ref.name_file_U_temp = [param_ref.name_file_U_temp ...
   %         {name_file_U_temp}];
    %    param.name_file_U_temp = param_ref.name_file_U_temp;
    
param.name_file_U_temp = cell(1,11);

for i = 1:11
   param.name_file_U_temp{i} = [ param.folder_data 'folder_file_temp_test_variance_tensor_2_modes__a_cst__decor_by_subsampl_bt_decor_choice_htgen2_threshold_1fct_test_b/' 'dsamp_11_' num2str(i) '_U_temp.mat' ]; 
end
    
bt_tot = ones(1,param.N_tot);

param = exportFields(param);

param = quadratic_variation_estimation(param,bt_tot);

load('/home/guillaume.lepape@eurogiciel.fr/HDD/REDLUM_CODE/data/diffusion_mode_test_variance_tensor_2_modes_a_cst_meth_htgen2.mat','z');
x=param.dX(1)*(0:(param.MX(1)-1));
y=param.dX(2)*(0:(param.MX(2)-1));
a = reshape(z,[param.MX 2 2]);
figure;imagesc(x,y,a(:,:,1,1)');axis xy; axis equal; colorbar;
figure;imagesc(x,y,a(:,:,1,2)');axis xy; axis equal; colorbar;
figure;imagesc(x,y,a(:,:,2,1)');axis xy; axis equal; colorbar;
figure;imagesc(x,y,a(:,:,2,2)');axis xy; axis equal; colorbar;

end

