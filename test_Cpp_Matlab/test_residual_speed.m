function test_residual_speed()

% load param
load('/home/guillaume.lepape@eurogiciel.fr/HDD/Guillaume_Le_Pape_data/REDLUM_CODE/podfs2/resultats/current_results/1stresult_DNS100_OpenFOAM_2D_2020_2_modes_htgen2b_fullsto/_integ_Ito_estim_rmv_fv_DFSPN.mat');

% set x and y arrays to plot figures
x = param.dX(1)*(0:param.MX(1)-1);
y = param.dX(2)*(0:param.MX(2)-1);

% load snapshots
load('/home/guillaume.lepape@eurogiciel.fr/HDD/Guillaume_Le_Pape_data/REDLUM_CODE/data/folder_DNS100_OpenFOAM_2D_2020/file_DNS100_OpenFOAM_2D_2020_run_1.mat');
U_1 = U(:,:,1,:);
figure;imagesc(x,y,U_1(:,:,1)');axis xy; axis equal; colorbar;
U_1 = reshape(U_1, [param.M param.d]);


%load the temporal mean
load([param.folder_data 'DNS100_OpenFOAM_2D_2020_blocks_truncated0_U_centered.mat']);

%squeeze it to delete dimension 2 equal to 1
m_U = squeeze(m_U); % size m_U : from M x 1 x d to M x d

% plot temporal mean
m_U_reshape = reshape(m_U, [param.MX param.d]);
figure;imagesc(x,y,m_U_reshape(:,:,1)');axis xy; axis equal; colorbar;
clear m_U_reshape;


% load spatial modes
load('/home/guillaume.lepape@eurogiciel.fr/HDD/Guillaume_Le_Pape_data/REDLUM_CODE/data/mode_DNS100_OpenFOAM_2D_2020_blocks_truncated_2_modes.mat');
% remove temporal mean from spatial modes
phi_U = phi_m_U(:,1:2,:);

% plot spatial modes
phi_U_reshape = reshape(phi_U,[param.MX 2 param.d]);
figure;imagesc(x,y,phi_U_reshape(:,:,1,1)');axis xy; axis equal; colorbar;
figure;imagesc(x,y,phi_U_reshape(:,:,2,1)');axis xy; axis equal; colorbar;
clear phi_U_reshape;

% load temporal modes
load('/home/guillaume.lepape@eurogiciel.fr/HDD/Guillaume_Le_Pape_data/REDLUM_CODE/podfs2/resultats/current_results/modes_bi_before_subsampling_DNS100_OpenFOAM_2D_2020_blocks_truncated_nb_modes_2.mat');

% plot temporal modes
plot(param.dt*(0:param.N_tot-1),bt(:,1), param.dt*(0:param.N_tot-1),bt(:,2));

% compute the normalized field
U_normalized = squeeze(U_1) - m_U;

% initialize reconstructed field
U_w = zeros(param.M,param.d);
U_w(:,1) = phi_U(:,:,1)*bt(1,:)'; 
U_w(:,2) = phi_U(:,:,2)*bt(1,:)'; 

% compute residual speed to the first time step
residual_speed_1 = U_normalized - U_w;

% plot magnitude, x and y components of residual speed field
residual_speed_1_reshape = reshape(residual_speed_1,[param.MX param.d]);
figure;imagesc(x,y,sqrt(residual_speed_1_reshape(:,:,1)'.*residual_speed_1_reshape(:,:,1)' + residual_speed_1_reshape(:,:,2)'.*residual_speed_1_reshape(:,:,2)'));axis xy; axis equal; colorbar;
figure;imagesc(x,y,residual_speed_1_reshape(:,:,1)');axis xy; axis equal; colorbar;
figure;imagesc(x,y,residual_speed_1_reshape(:,:,2)');axis xy; axis equal; colorbar;


end

