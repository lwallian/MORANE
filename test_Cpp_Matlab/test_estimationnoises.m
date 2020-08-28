function test_estimationnoises()

global estim_rmv_fv;
global stochastic_integration;
estim_rmv_fv = 0;
stochastic_integration = 'Ito';

param.nb_modes = 2;
m = param.nb_modes;
n = param.nb_modes;
param.dt = 0.05;
param.decor_by_subsampl.choice_n_subsample = 'htgen2';
param.folder_data = '../data/';
param.eq_proj_div_free = 0;
param.big_data = 1;
param.d = 2;
param.N_tot = 500;
param.T = param.dt*(param.N_tot-1);
lambda =  [ 3298.90006327741548375343 3218.50831458436823595548 ];
lambda = lambda/param.N_tot;
param.lambda = lambda;
param.replication_data = false;
param.viscosity = 0.01;
param.data_in_blocks.bool = 1;
param.decor_by_subsampl.meth = 'bt_decor';
param.decor_by_subsampl.spectrum_threshold = 1;
param.decor_by_subsampl.test_fct = 'b';
param.decor_by_subsampl.bool = false;
dependance_on_time_of_a = '_a_cst_';
char_filter = [];
threshold_str = num2str(param.decor_by_subsampl.spectrum_threshold);
iii = (threshold_str=='.');
threshold_str(iii)='_';

%bt = zeros(param.N_tot,param.nb_modes);
%file_temporalModes = [ '~/HDD/flow_past_cylinder/donnees_parallele/ITHACAoutput/temporalModes_' num2str(param.nb_modes) 'modes/U_mat.txt' ];
%fileID = fopen( file_temporalModes );
%C = textscan(fileID, '%f %f %f %f %f %f %f %f %f' );

%disp("Read the temporal modes");

%for i = 1:size(bt,1)
%    for j = 1:size(bt,2)
%        bt(i,j) = C{j}(i);
%    end
%end

%% Compute bt and deriv_bt

[bt, deriv_bt] = compute_bt(param);


%% Compute error for original mesh
param.MX = [251 126];
param.M = param.MX(1)*param.MX(2);
param.dX = [0.008 0.008];
param.type_data = 'test_estimationnoises_refine1';
param.name_file_mode = [ param.folder_data 'mode_' param.type_data '_' num2str(param.nb_modes) '_modes.mat'];
param.folder_file_U_temp = ...
        [ param.folder_data 'folder_file_temp_' param.type_data '_' ...
        num2str(param.nb_modes) '_modes_' ...
        dependance_on_time_of_a char_filter ...
        '_decor_by_subsampl_' num2str(param.decor_by_subsampl.meth) ...
        '_choice_' num2str(param.decor_by_subsampl.choice_n_subsample)  ...
        '_threshold_' threshold_str  ...
        'fct_test_' param.decor_by_subsampl.test_fct '/' ];

for i = 1:5
   param.name_file_U_temp{i} = [ param.folder_file_U_temp 'dsamp_3_' num2str(i) '_U_temp' ]; 
end

I_deter = zeros(n,1);
L_deter = zeros(n,n);
C_deter = zeros(n,n,n);
deter = struct('I',I_deter,'L',L_deter,'C',C_deter);

I_sto = zeros(n,1);
L_sto = zeros(n,n);
C_sto = zeros(n,n,n);
sto = struct('I',I_sto,'L',L_sto,'C',C_sto);

I_sto = I_deter + I_sto;
L_sto = L_deter + L_sto;
C_sto = C_deter + C_sto;
tot = struct('I',I_sto,'L',L_sto,'C',C_sto);
ILC=struct('deter',deter,'sto',sto,'tot',tot);

disp("Export residual speed");
[residualSpeed] = exportResidualSpeed(param,deriv_bt);

disp("Export spatial modes");
[~] = exportSpatialMode(param,bt,residualSpeed);
clear residualSpeed;

disp("Error for the original mesh")

%% Read matrice noises computed with C++ code
refine = 1;
[theta_theta, alpha_theta, theta_theta] = read_matriceNoises(param,refine);

%% Run matlab noises matrix computation

[result,pseudo_chol] = estimation_noises(param,bt,ILC);

fprintf('\n');

%% Compute error for the first refinement
param.MX = [501 251];
param.M = param.MX(1)*param.MX(2);
param.dX = [0.004 0.004];
param.type_data = 'test_estimationnoises_refine2';
param.name_file_mode = [ param.folder_data 'mode_' param.type_data '_' num2str(param.nb_modes) '_modes.mat'];
param.folder_file_U_temp = ...
        [ param.folder_data 'folder_file_temp_' param.type_data '_' ...
        num2str(param.nb_modes) '_modes_' ...
        dependance_on_time_of_a char_filter ...
        '_decor_by_subsampl_' num2str(param.decor_by_subsampl.meth) ...
        '_choice_' num2str(param.decor_by_subsampl.choice_n_subsample)  ...
        '_threshold_' threshold_str  ...
        'fct_test_' param.decor_by_subsampl.test_fct '/' ];

for i = 1:5
   param.name_file_U_temp{i} = [ param.folder_file_U_temp 'dsamp_3_' num2str(i) '_U_temp' ]; 
end

I_deter = zeros(n,1);
L_deter = zeros(n,n);
C_deter = zeros(n,n,n);
deter = struct('I',I_deter,'L',L_deter,'C',C_deter);

I_sto = zeros(n,1);
L_sto = zeros(n,n);
C_sto = zeros(n,n,n);
sto = struct('I',I_sto,'L',L_sto,'C',C_sto);

I_sto = I_deter + I_sto;
L_sto = L_deter + L_sto;
C_sto = C_deter + C_sto;
tot = struct('I',I_sto,'L',L_sto,'C',C_sto);
ILC=struct('deter',deter,'sto',sto,'tot',tot);

disp("Export residual speed");
residualSpeed = exportResidualSpeed(param,deriv_bt);

disp("Export spatial modes");
[~] = exportSpatialMode(param,bt,residualSpeed);
clear residualSpeed;

disp("Error for the first refinement")

%% Read matrice noises computed with C++ code
refine = 2;
[theta_theta, alpha_theta, theta_theta] = read_matriceNoises(param,refine);

%% Run matlab noises matrix computation

[result,pseudo_chol] = estimation_noises(param,bt,ILC);

fprintf('\n');


%% Compute error for second refinement
param.MX = [1001 501];
param.M = param.MX(1)*param.MX(2);
param.dX = [0.002 0.002];
param.type_data = 'test_estimationnoises_refine4';
param.name_file_mode = [ param.folder_data 'mode_' param.type_data '_' num2str(param.nb_modes) '_modes.mat'];
param.folder_file_U_temp = ...
        [ param.folder_data 'folder_file_temp_' param.type_data '_' ...
        num2str(param.nb_modes) '_modes_' ...
        dependance_on_time_of_a char_filter ...
        '_decor_by_subsampl_' num2str(param.decor_by_subsampl.meth) ...
        '_choice_' num2str(param.decor_by_subsampl.choice_n_subsample)  ...
        '_threshold_' threshold_str  ...
        'fct_test_' param.decor_by_subsampl.test_fct '/' ];

for i = 1:5
   param.name_file_U_temp{i} = [ param.folder_file_U_temp 'dsamp_3_' num2str(i) '_U_temp' ]; 
end

I_deter = zeros(n,1);
L_deter = zeros(n,n);
C_deter = zeros(n,n,n);
deter = struct('I',I_deter,'L',L_deter,'C',C_deter);

I_sto = zeros(n,1);
L_sto = zeros(n,n);
C_sto = zeros(n,n,n);
sto = struct('I',I_sto,'L',L_sto,'C',C_sto);

I_sto = I_deter + I_sto;
L_sto = L_deter + L_sto;
C_sto = C_deter + C_sto;
tot = struct('I',I_sto,'L',L_sto,'C',C_sto);
ILC=struct('deter',deter,'sto',sto,'tot',tot);

disp("Export residual speed");
[residualSpeed] = exportResidualSpeed(param,deriv_bt);

disp("Export spatial modes");
[~] = exportSpatialMode(param,bt,residualSpeed);
clear residualSpeed;

disp("Error for the second refinement")

%% Read matrice noises computed with C++ code
refine = 4;
[theta_theta, alpha_theta, theta_theta] = read_matriceNoises(param,refine);

%% Run matlab noises matrix computation

[result,pseudo_chol] = estimation_noises(param,bt,ILC);

fprintf('\n');

%% Compute error for third refinement
%{
param.MX = [2001 1001];
param.M = param.MX(1)*param.MX(2);
param.dX = [0.001 0.001];
param.type_data = 'test_estimationnoises_refine8';
param.name_file_mode = [ param.folder_data 'mode_' param.type_data '_' num2str(param.nb_modes) '_modes.mat'];
param.folder_file_U_temp = ...
        [ param.folder_data 'folder_file_temp_' param.type_data '_' ...
        num2str(param.nb_modes) '_modes_' ...
        dependance_on_time_of_a char_filter ...
        '_decor_by_subsampl_' num2str(param.decor_by_subsampl.meth) ...
        '_choice_' num2str(param.decor_by_subsampl.choice_n_subsample)  ...
        '_threshold_' threshold_str  ...
        'fct_test_' param.decor_by_subsampl.test_fct '/' ];

for i = 1:5
   param.name_file_U_temp{i} = [ param.folder_file_U_temp 'dsamp_3_' num2str(i) '_U_temp' ]; 
end

I_deter = zeros(n,1);
L_deter = zeros(n,n);
C_deter = zeros(n,n,n);
deter = struct('I',I_deter,'L',L_deter,'C',C_deter);

I_sto = zeros(n,1);
L_sto = zeros(n,n);
C_sto = zeros(n,n,n);
sto = struct('I',I_sto,'L',L_sto,'C',C_sto);

I_sto = I_deter + I_sto;
L_sto = L_deter + L_sto;
C_sto = C_deter + C_sto;
tot = struct('I',I_sto,'L',L_sto,'C',C_sto);
ILC=struct('deter',deter,'sto',sto,'tot',tot);

disp("Export residual speed");
[residualSpeed] = exportResidualSpeed(param,deriv_bt);

disp("Export spatial modes");
[~] = exportSpatialMode(param,bt,residualSpeed);
clear residualSpeed;

disp("Error for the third refinement")

[result,pseudo_chol] = estimation_noises(param,bt,ILC);

fprintf('\n');
%}

end

function [bt, deriv_bt] = compute_bt(param)
    bt = zeros(param.N_tot, param.nb_modes);
    deriv_bt = zeros(param.N_tot, param.nb_modes);
    
    A = [ 2.5 2. ];
    phi = [ 1. 2. ];
    period = 5;
    
    for k = 1:param.N_tot
        t = (k-1)*param.dt;
        
        for i = 1:param.nb_modes
            bt(k,i) = A(i)*sin(2*pi*t/period + phi(i));
            deriv_bt(k,i) = A(i)*(2*pi/period)*cos(2*pi*t/period + phi(i));
        end
    
    end
    
end


function [h] = compute_h(param)
    h = zeros(param.M,param.d);
    
    for i = 1:param.MX(1)
        for j = 1:param.MX(2)
            x = (i-1)*param.dX(1);
            y = (j-1)*param.dX(2);
            h(i + param.MX(1)*(j-1),1) = 100*( cos( x*(x-2)*y*(y-1) ) - 1 );
            h(i + param.MX(1)*(j-1),2) = 100*2*y*(y-1)*sin(2*x)*x*(x-2) ;
        end
    end
end

function [s] = compute_s(deriv_bt)  
    s = sum(deriv_bt,2);
end

function [residualSpeed] = exportResidualSpeed(param,deriv_bt)
    h = compute_h(param);
    s = compute_s(deriv_bt);
    
    residualSpeed = zeros(param.M, param.N_tot, param.d);
    
    for i = 1:param.N_tot
        residualSpeed(:,i,:) = h*s(i,1);
    end
    
    mkdir( param.folder_file_U_temp );
    
    for i = 1:5
        Q = 100;
        U = residualSpeed(:,(i-1)*Q+1:i*Q,:);
        save( param.name_file_U_temp{i},'U','-v7.3');
    end
end

function [mean] = compute_mean_w(param,phi_U,bt)
    mean = zeros(param.M,param.d);
    
    for i = 1:size(phi_U,2)
        S = 0;
        
        for k = 1:param.N_tot
            S = S + bt(k,i);
        end
        
        S = S/param.N_tot;
        
        mean = mean + S*squeeze(phi_U(:,i,:));
    end
    %{
    x = param.dX(1)*(0:param.MX(1)-1);
    y = param.dX(2)*(0:param.MX(2)-1);
    
    mean_save = reshape(mean, [param.MX param.d]);
    
    figure;imagesc(x,y,mean_save(:,:,1)');axis xy; axis equal; colorbar;
    figure;imagesc(x,y,mean_save(:,:,2)');axis xy; axis equal; colorbar;
    %}
end

function [mean] = compute_mean_vprime(param,residualSpeed)
    mean = zeros(param.M,param.d);
    
    for k = 1:param.N_tot
        mean = mean + squeeze(residualSpeed(:,k,:));
    end
    
    mean = mean / param.N_tot;
    %{
    x = param.dX(1)*(0:param.MX(1)-1);
    y = param.dX(2)*(0:param.MX(2)-1);
    
    mean_save = reshape(mean, [param.MX param.d]);
    
    figure;imagesc(x,y,mean_save(:,:,1)');axis xy; axis equal; colorbar;
    figure;imagesc(x,y,mean_save(:,:,2)');axis xy; axis equal; colorbar;
    %}
end

function [phi_U] = compute_spatialModes(param)
    phi_U = zeros(param.M,param.nb_modes,param.d);
    
    for k = 1:param.nb_modes 
        for i = 1:param.MX(1)
            for j = 1:param.MX(2)
                x = (i-1)*param.dX(1);
                y = (j-1)*param.dX(2);
                phi_U(i + param.MX(1)*(j-1),k,1) = 10*x*(x-2)*y*(y-1)*exp(-x*x*(k-1));
                phi_U(i + param.MX(1)*(j-1),k,2) = 10*x*(x-2)*y*(y-1)*exp(-y*y*(k-1));
            end
        end
    end
end

function [phi_m_U] = exportSpatialMode(param,bt,residualSpeed)
    phi_U = compute_spatialModes(param);
    mean_w_U = compute_mean_w(param,phi_U,bt);
    mean_v_prime_U = compute_mean_vprime(param,residualSpeed);
   
    phi_m_U = zeros(param.M,param.nb_modes+1,param.d);
    phi_m_U(:,1:param.nb_modes,:) = phi_U;
    phi_m_U(:,param.nb_modes+1,:) = mean_w_U + mean_v_prime_U;
    
    %{
    x = param.dX(1)*(0:param.MX(1)-1);
    y = param.dX(2)*(0:param.MX(2)-1);
 
    mean_w_U_export = reshape(mean_w_U,[param.MX param.d]);
    figure;imagesc(x,y,mean_w_U_export(:,:,1)');axis xy; axis equal; colorbar;
    figure;imagesc(x,y,mean_w_U_export(:,:,2)');axis xy; axis equal; colorbar;
 
    mean_vprime_U_export = reshape(mean_v_prime_U,[param.MX param.d]);
    figure;imagesc(x,y,mean_vprime_U_export(:,:,1)');axis xy; axis equal; colorbar;
    figure;imagesc(x,y,mean_vprime_U_export(:,:,2)');axis xy; axis equal; colorbar;
    %}
    save(param.name_file_mode,'phi_m_U','-v7.3');
end




function [theta_theta,alpha_theta,alpha_alpha] = read_matriceNoises(param,refine)

nb_modes = param.nb_modes;
nb_modes

%% read theta_theta
file_theta_theta = ['~/HDD/Guillaume_Le_Pape_data/unitTestITHACALUM/noisesMatrix_spatial_dependency/noisesMatrix_refine' ...
    num2str(refine) '/ITHACAoutput/theta_theta_mat.txt'];

fileID = fopen( file_theta_theta );
C = textscan( fileID, '%f %f %f %f %f %f %f %f %f' );
    
theta_theta = zeros(nb_modes, nb_modes);

for i = 1:nb_modes
    for j = 1:nb_modes
       theta_theta(i,j) = C{j}(i);
    end
end

%% read alpha_theta
file_alpha_theta = ['../../unitTestITHACALUM/noisesMatrix_spatial_dependency/noisesMatrix_refine' ...
    num2str(refine) '/ITHACAoutput/alpha_theta_mat.txt'];

fileID = fopen( file_alpha_theta );
C = textscan(fileID, '%f %f %f %f %f %f %f %f %f' );
    
alpha_theta = zeros(nb_modes*nb_modes, nb_modes);

for i = 1:(nb_modes*nb_modes)
  for j = 1:nb_modes
      alpha_theta(i,j) = C{j}(i);
   end
end
    
%% read alpha_alpha

file_alpha_alpha = ['../../unitTestITHACALUM/noisesMatrix_spatial_dependency/noisesMatrix_refine' ...
    num2str(refine) '/ITHACAoutput/alpha_alpha_mat.txt'];

fileID = fopen( file_alpha_alpha );
C = textscan(fileID, '%f %f %f %f %f %f %f %f %f' );
    
alpha_alpha = zeros(nb_modes*nb_modes, nb_modes*nb_modes);

for i = 1:(nb_modes*nb_modes)
  for j = 1:(nb_modes*nb_modes)
      alpha_alpha(i,j) = C{j}(i);
   end
end

end