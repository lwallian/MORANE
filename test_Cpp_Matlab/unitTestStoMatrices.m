function test_spatialModes2D_sto()

%% set parameters that cannot change after refinement
param.nb_modes = 9;
param.d = 2;
param.lambda = zeros(8,1);
param.big_data = true;
param.eq_proj_div_free = 0;
param.a_time_dependant = 0;
param.viscosity = 0.01;

param.folder_data = [ pwd '/../data' ];

error_L = zeros(4,1);
error_S = zeros(4,1);
dX = [0.008 0.004 0.002 0.001];

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

param.type_data = 'modes_test_sto_refine1.mat';
param.name_file_mode = [ param.folder_data param.type_data ];

% Set spatial modes
phi_m_U = zeros(param.M,param.nb_modes,param.d);

for i = 1:param.MX(1)
    for j = 1:param.MX(2)
        x = (i-1)*param.dX(1);
        y = (j-1)*param.dX(2);
        for k = 1:param.nb_modes-1
            phi_m_U(i + param.MX(1)*(j-1),k,:) = [ x*(x-2)*y*(y-1)*exp(-(k-1)*x^2)  x*(x-2)*y*(y-1)*exp(-(k-1)*y^2) ];
        end
    end
end

% Save them
save(param.name_file_mode,'phi_m_U','-v7.3');


% set variance tensor a
z = zeros(param.M,1,param.d,param.d);

param.name_file_diffusion_mode = [ param.folder_data 'a_test_sto1' ];

for i = 1:param.MX(1)
    for j = 1:param.MX(2)
        x = (i-1)*param.dX(1);
        y = (j-1)*param.dX(2);
        
        % xx-component
        z(i + param.MX(1)*(j-1),1,1,1) = x*(x-2)*y*(y-1)*cos(x*y);
        % xy-component 
        z(i + param.MX(1)*(j-1),1,1,2) = x*(x-2)*y*(y-1)/(1 + x*y);
        % yx-component
        z(i + param.MX(1)*(j-1),1,2,1) = z(i + param.MX(1)*(j-1),1,1,2);
        % yy-component
        z(i + param.MX(1)*(j-1),1,2,2) = x*(x-2)*y*(y-1)*sin(x*y);
    end
end

% save it
save(param.name_file_diffusion_mode,'z','-v7.3');

%% read C++ matrices
refine = 1;
[L_Cpp_1,S_Cpp_1] = read_Cpp_sto(param,refine); 

%% compute the matlab deter matrices

[F1_1,F2_1] = coefficients_sto(param);

F1_1 = F1_1(1:param.nb_modes-1,1:param.nb_modes-1);
F2_1 = F2_1(1:param.nb_modes-1,1:param.nb_modes-1);

%% Compute the errors

% Error for L
error_L(1,1) = norm(L_Cpp_1 - F2_1,2)/norm(F2_1,2);
fprintf('Error is equal for L : %0.12f \n',error_L(1,1));

% Error for S
error_S(1,1) = norm(S_Cpp_1 + F1_1,2)/norm(F1_1,2);
fprintf('Error is equal for S : %0.12f \n',error_S(1,1));
fprintf('\n');



%% first refinement
disp("Computing the error for the first refinement");

param.MX = [501 251];
param.M = param.MX(1) * param.MX(2);
param.dX = [0.004 0.004];

% set grid
gridx = zeros(param.MX(1),1);
gridy = zeros(param.MX(2),1);

grid = cell(1,2);

grid{1} = gridx;
grid{2} = gridy;

param.grid = grid;

param.type_data = 'modes_test_sto_refine2.mat';
param.name_file_mode = [ param.folder_data param.type_data ];

% Set spatial modes
phi_m_U = zeros(param.M,param.nb_modes,param.d);

for i = 1:param.MX(1)
    for j = 1:param.MX(2)
        x = (i-1)*param.dX(1);
        y = (j-1)*param.dX(2);
        for k = 1:param.nb_modes-1
            phi_m_U(i + param.MX(1)*(j-1),k,:) = [ x*(x-2)*y*(y-1)*exp(-(k-1)*x^2)  x*(x-2)*y*(y-1)*exp(-(k-1)*y^2) ];
        end
    end
end

% Save them
save(param.name_file_mode,'phi_m_U','-v7.3');


% set variance tensor a
z = zeros(param.M,1,param.d,param.d);

param.name_file_diffusion_mode = [ param.folder_data 'a_test_sto2' ];

for i = 1:param.MX(1)
    for j = 1:param.MX(2)
        x = (i-1)*param.dX(1);
        y = (j-1)*param.dX(2);
        
        % xx-component
        z(i + param.MX(1)*(j-1),1,1,1) = x*(x-2)*y*(y-1)*cos(x*y);
        % xy-component 
        z(i + param.MX(1)*(j-1),1,1,2) = x*(x-2)*y*(y-1)/(1 + x*y);
        % yx-component
        z(i + param.MX(1)*(j-1),1,2,1) = z(i + param.MX(1)*(j-1),1,1,2);
        % yy-component
        z(i + param.MX(1)*(j-1),1,2,2) = x*(x-2)*y*(y-1)*sin(x*y);
    end
end

% save it
save(param.name_file_diffusion_mode,'z','-v7.3');

%% read C++ matrices
refine = 2;
[L_Cpp_2,S_Cpp_2] = read_Cpp_sto(param,refine); 

%% compute the matlab deter matrices

[F1_2,F2_2] = coefficients_sto(param);

F1_2 = F1_2(1:param.nb_modes-1,1:param.nb_modes-1);
F2_2 = F2_2(1:param.nb_modes-1,1:param.nb_modes-1);

%% Compute the errors

% Error for L
error_L(2,1) = norm(L_Cpp_2 - F2_2,2)/norm(F2_2,2);
fprintf('Error is equal for L : %0.12f \n',error_L(2,1));

% Error for S
error_S(2,1) = norm(S_Cpp_2 + F1_2,2)/norm(F1_2,2);
fprintf('Error is equal for S : %0.12f \n',error_S(2,1));
fprintf('\n');


%% second refinement
disp("Computing the error for the second refinement");

param.MX = [1001 501];
param.M = param.MX(1) * param.MX(2);
param.dX = [0.002 0.002];

% set grid
gridx = zeros(param.MX(1),1);
gridy = zeros(param.MX(2),1);

grid = cell(1,2);

grid{1} = gridx;
grid{2} = gridy;

param.grid = grid;

param.type_data = 'modes_test_sto_refine4.mat';
param.name_file_mode = [ param.folder_data param.type_data ];

% Set spatial modes
phi_m_U = zeros(param.M,param.nb_modes,param.d);

for i = 1:param.MX(1)
    for j = 1:param.MX(2)
        x = (i-1)*param.dX(1);
        y = (j-1)*param.dX(2);
        for k = 1:param.nb_modes-1
            phi_m_U(i + param.MX(1)*(j-1),k,:) = [ x*(x-2)*y*(y-1)*exp(-(k-1)*x^2)  x*(x-2)*y*(y-1)*exp(-(k-1)*y^2) ];
        end
    end
end

% Save them
save(param.name_file_mode,'phi_m_U','-v7.3');


% set variance tensor a
z = zeros(param.M,1,param.d,param.d);

param.name_file_diffusion_mode = [ param.folder_data 'a_test_sto4' ];

for i = 1:param.MX(1)
    for j = 1:param.MX(2)
        x = (i-1)*param.dX(1);
        y = (j-1)*param.dX(2);
        
        % xx-component
        z(i + param.MX(1)*(j-1),1,1,1) = x*(x-2)*y*(y-1)*cos(x*y);
        % xy-component 
        z(i + param.MX(1)*(j-1),1,1,2) = x*(x-2)*y*(y-1)/(1 + x*y);
        % yx-component
        z(i + param.MX(1)*(j-1),1,2,1) = z(i + param.MX(1)*(j-1),1,1,2);
        % yy-component
        z(i + param.MX(1)*(j-1),1,2,2) = x*(x-2)*y*(y-1)*sin(x*y);
    end
end

% save it
save(param.name_file_diffusion_mode,'z','-v7.3');

%% read C++ matrices
refine = 4;
[L_Cpp_4,S_Cpp_4] = read_Cpp_sto(param,refine); 

%% compute the matlab deter matrices

[F1_4,F2_4] = coefficients_sto(param);

F1_4 = F1_4(1:param.nb_modes-1,1:param.nb_modes-1);
F2_4 = F2_4(1:param.nb_modes-1,1:param.nb_modes-1);

%% Compute the errors

% Error for L
error_L(3,1) = norm(L_Cpp_4 - F2_4,2)/norm(F2_4,2);
fprintf('Error is equal for L : %0.12f \n',error_L(3,1));

% Error for S
error_S(3,1) = norm(S_Cpp_4 + F1_4,2)/norm(F1_4,2);
fprintf('Error is equal for S : %0.12f \n',error_S(3,1));
fprintf('\n');

%% third refinement
disp("Computing the error for the third refinement");

param.MX = [2001 1001];
param.M = param.MX(1) * param.MX(2);
param.dX = [0.001 0.001];

% set grid
gridx = zeros(param.MX(1),1);
gridy = zeros(param.MX(2),1);

grid = cell(1,2);

grid{1} = gridx;
grid{2} = gridy;

param.grid = grid;

param.type_data = 'modes_test_sto_refine8.mat';
param.name_file_mode = [ param.folder_data param.type_data ];

% Set spatial modes
phi_m_U = zeros(param.M,param.nb_modes,param.d);

for i = 1:param.MX(1)
    for j = 1:param.MX(2)
        x = (i-1)*param.dX(1);
        y = (j-1)*param.dX(2);
        for k = 1:param.nb_modes-1
            phi_m_U(i + param.MX(1)*(j-1),k,:) = [ x*(x-2)*y*(y-1)*exp(-(k-1)*x^2)  x*(x-2)*y*(y-1)*exp(-(k-1)*y^2) ];
        end
    end
end

% Save them
save(param.name_file_mode,'phi_m_U','-v7.3');


% set variance tensor a
z = zeros(param.M,1,param.d,param.d);

param.name_file_diffusion_mode = [ param.folder_data 'a_test_sto8' ];

for i = 1:param.MX(1)
    for j = 1:param.MX(2)
        x = (i-1)*param.dX(1);
        y = (j-1)*param.dX(2);
        
        % xx-component
        z(i + param.MX(1)*(j-1),1,1,1) = x*(x-2)*y*(y-1)*cos(x*y);
        % xy-component 
        z(i + param.MX(1)*(j-1),1,1,2) = x*(x-2)*y*(y-1)/(1 + x*y);
        % yx-component
        z(i + param.MX(1)*(j-1),1,2,1) = z(i + param.MX(1)*(j-1),1,1,2);
        % yy-component
        z(i + param.MX(1)*(j-1),1,2,2) = x*(x-2)*y*(y-1)*sin(x*y);
    end
end

% save it
save(param.name_file_diffusion_mode,'z','-v7.3');

%% read C++ matrices
refine = 8;
[L_Cpp_8,S_Cpp_8] = read_Cpp_sto(param,refine); 

%% compute the matlab deter matrices

[F1_8,F2_8] = coefficients_sto(param);

F1_8 = F1_8(1:param.nb_modes-1,1:param.nb_modes-1);
F2_8 = F2_8(1:param.nb_modes-1,1:param.nb_modes-1);

%% Compute the errors

% Error for L
error_L(4,1) = norm(L_Cpp_8 - F2_8,2)/norm(F2_8,2);
fprintf('Error is equal for L : %0.12f \n',error_L(4,1));

% Error for S
error_S(4,1) = norm(S_Cpp_8 + F1_8,2)/norm(F1_8,2);
fprintf('Error is equal for S : %0.12f \n',error_S(4,1));
fprintf('\n');

%% Plot the errors

loglog(1 ./ dX, error_L, 1 ./ dX, error_S)

end
