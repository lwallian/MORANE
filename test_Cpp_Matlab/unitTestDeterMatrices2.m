function unitTestDeterMatrices2()

%% set parameters that cannot change after refinement
param.viscosity = 0.01;
param.lambda = zeros(8,1);
param.big_data = true;
param.eq_proj_div_free = 0;
param.nb_modes = 9;
param.d = 2;

param.folder_data = [ pwd '/../data' ];

error_L = zeros(3,1);
error_C = zeros(3,1);
dX = [0.008 0.004 0.002];

%% original mesh
disp("Computing error for the original mesh");

param.MX = [251 126];
param.M = param.MX(1) * param.MX(2);
param.dX = [0.008 0.008];

% set grid
gridx = zeros(param.MX(1),1);
gridy = zeros(param.MX(2),1);
grid = cell(1,2);
grid{1} = gridx;
grid{2} = gridy;
param.grid = grid;

param.type_data = 'modes_test_deter_refine1.mat';
param.name_file_mode = [ param.folder_data param.type_data];

U = zeros(param.M,param.nb_modes,param.d);

for k = 1:param.nb_modes-1
    for i = 1:param.MX(1)
        for j = 1:param.MX(2)
            x = (i-1)*param.dX(1);
            y = (j-1)*param.dX(2);
            U(i + param.MX(1)*(j-1),k,:) = [ x*(x-2)*y*(y-1)*exp(-(k-1)*x^2)  x*(x-2)*y*(y-1)*exp(-(k-1)*y^2)];
        end
    end
end

phi_m_U = U;
clear U;
save(param.name_file_mode,'phi_m_U','-v7.3');

%% read C++ matrices
refine = 1;
[B_Cpp_1,C_Cpp_1] = read_Cpp_deter(param,refine); 

%% compute the matlab deter matrices
[~,L_deter_1,C_deter_1,param] = param_ODE_bt_deter(param.name_file_mode, param, param.grid);
C_deter_1 = permute(C_deter_1,[2 3 1]);

%% Compute the error
% Error for L matrix
error_L(1,1) = norm(B_Cpp_1 + L_deter_1,2)/norm(L_deter_1,2);
fprintf('Error is equal for L : %0.12f \n',error_L(1,1));

% Error for C matrix
for i = param.nb_modes-1
    error_C(1,1) = error_C(1,1) + norm(C_Cpp_1(:,:,i) + C_deter_1(:,:,i),2)/norm(C_deter_1(:,:,i),2);
end

error_C(1,1) = error_C(1,1)/( param.nb_modes-1 );

fprintf('Error is equal for C : %0.12f \n',error_C(1,1));
fprintf('\n');

%% first refinement
disp("Computing the error for the first refinement");

param.MX = [ 501 251 ];
param.M = param.MX(1) * param.MX(2);
param.dX = [0.004 0.004];

% set grid
gridx = zeros(param.MX(1),1);
gridy = zeros(param.MX(2),1);
grid = cell(1,2);
grid{1} = gridx;
grid{2} = gridy;
param.grid = grid;

param.type_data = 'modes_test_deter_refine2.mat';
param.name_file_mode = [ param.folder_data param.type_data];

U = zeros(param.M,param.nb_modes,param.d);

for k = 1:param.nb_modes-1
    for i = 1:param.MX(1)
        for j = 1:param.MX(2)
            x = (i-1)*param.dX(1);
            y = (j-1)*param.dX(2);
            U(i + param.MX(1)*(j-1),k,:) = [ x*(x-2)*y*(y-1)*exp(-(k-1)*x^2)  x*(x-2)*y*(y-1)*exp(-(k-1)*y^2)];
        end
    end
end

phi_m_U = U;
clear U;
save([param.folder_data param.type_data],'phi_m_U','-v7.3');

%% read C++ matrices
refine = 2;
[B_Cpp_2,C_Cpp_2] = read_Cpp_deter(param,refine);

%% compute the matlab deter matrices
[~,L_deter_2,C_deter_2,param] = param_ODE_bt_deter(param.name_file_mode, param, param.grid);
C_deter_2 = permute(C_deter_2,[2 3 1]);

%% Compute the error

% Error for L
error_L(2,1) = norm(B_Cpp_2 + L_deter_2,2)/norm(L_deter_2,2);
fprintf('Error is equal for L : %0.12f \n', error_L(2,1) );

% Error for C
for i = param.nb_modes-1
    error_C(2,1) = error_C(2,1) + norm(C_Cpp_2(:,:,i) + C_deter_2(:,:,i),2)/norm(C_deter_2(:,:,i),2);
end

error_C(2,1) = error_C(2,1)/( param.nb_modes-1 );

fprintf('Error is equal for C : %0.12f \n',error_C(2,1));
fprintf('\n');

%% second refinement
disp("Computing the error for the second refinement");

param.MX = [ 1001 501 ];
param.M = param.MX(1) * param.MX(2);
param.dX = [0.002 0.002];

% set grid
gridx = zeros(param.MX(1),1);
gridy = zeros(param.MX(2),1);
grid = cell(1,2);
grid{1} = gridx;
grid{2} = gridy;
param.grid = grid;

param.type_data = 'modes_test_deter_refine4.mat';
param.name_file_mode = [ param.folder_data param.type_data ];

U = zeros(param.M,param.nb_modes,param.d);

for k = 1:param.nb_modes-1
    for i = 1:param.MX(1)
        for j = 1:param.MX(2)
            x = (i-1)*param.dX(1);
            y = (j-1)*param.dX(2);
            U(i + param.MX(1)*(j-1),k,:) = [ x*(x-2)*y*(y-1)*exp(-(k-1)*x^2)  x*(x-2)*y*(y-1)*exp(-(k-1)*y^2)];
        end
    end
end

phi_m_U = U;
clear U;
save([param.folder_data param.type_data],'phi_m_U','-v7.3');

%% read C++ matrices
refine = 4;
[B_Cpp_4,C_Cpp_4] = read_Cpp_deter(param,refine); 

%% compute the matlab deter matrices
[~,L_deter_4,C_deter_4,param] = param_ODE_bt_deter(param.name_file_mode, param, param.grid);
C_deter_4 = permute(C_deter_4,[2 3 1]);

%% Compute the errors

% Error for L
error_L(3,1) = norm(B_Cpp_4 + L_deter_4,2)/norm(L_deter_4,2);
fprintf('Error is equal for L : %0.12f \n',error_L(3,1));

% Error for C

for i = param.nb_modes-1
    error_C(3,1) = error_C(3,1) + norm(C_Cpp_4(:,:,i) + C_deter_4(:,:,i),2)/norm(C_deter_4(:,:,i),2) ;
end

error_C(3,1) = error_C(3,1)/(param.nb_modes-1) ;
fprintf('Error is equal for C : %0.12f \n',error_C(3,1));
fprintf('\n');

%% Plot the error

dX 
error_L
error_C

loglog(1 ./ dX, error_L, 1 ./ dX, error_C)

end
