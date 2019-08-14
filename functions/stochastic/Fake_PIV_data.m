function Fake_PIV_data(param)
% Variance tensor estimation
%

%% Get parameters

switch param.type_data
    case {'DNS100_inc3d_2D_2018_11_16_blocks',...
            'DNS100_inc3d_2D_2018_11_16_blocks_truncated',...
            'DNS100_inc3d_2D_2018_11_16_blocks_test_basis'}
        
        % Smoothing
        %         new_distance = 0.041666666666666664;
        std_space = 0.203125;
        
        % Number of velocity component
        d_PIV = 2;
        
        % Slicing
        %         z_slice = 30;
        
        % XP param
        u_inf_measured = 0.135; % m/s
        cil_diameter = 12; % 12mm
        X_cyl = [ -75.60 0.75 ];
        
        % The dimensions of PIV and DNS are not the same,
        % therefore we must find the same number of sampled points
        % for each dimension
        % PIV_range ---->  x = (0.74,10.44) y=(-2.84,2.83)
        % DNS_range ---->  x = (-2.5,15.04) y=(-1.95,1.95)
        PIV_range = [0.74 10.44 ; -2.84 2.83 ] ;
        x_DNS = param.dX(1) * (0:(param.MX(1)-1));
        x_DNS = x_DNS - 5;
        y_DNS = param.dX(2) * (0:(param.MX(2)-1));
        y_DNS = y_DNS - mean(y_DNS);
        DNS_range = [x_DNS([1 end]) ; y_DNS([1 end]) ] ;
        %         DNS_range = [-2.5 15.04 ; -1.95 1.95 ] ;
        PIV_range = PIV_range + (-3) * ...
            [ param.dX(1) -param.dX(1) ; param.dX(2) -param.dX(2)] ;
        %         DNS_range = DNS_range + (-3) * ...
        %             [ param.dX(1) -param.dX(1) ; param.dX(2) -param.dX(2)] ;
        
        filename = [param.folder_data ...
            '\XP_Irstea\wake_Re100_export_190710_4107\' ...
            'B0001.dat'];
        
        error_estim = 6e-2;
        
    case {'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks',...
            'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_truncated',...
            'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_test_basis'}
        
        % Smoothing
        new_distance = 0.041666666666666664;
        std_space = 0.203125;
        
        % Number of velocity component
        d_PIV = 2;
        
        % Slicing
        z_slice = 30;
        
        % XP param
        u_inf_measured = 0.388; % m/s
        cil_diameter = 12; % 12mm
        X_cyl = [ -75.60 0.75 ];
        
        % The dimensions of PIV and DNS are not the same,
        % therefore we must find the same number of sampled points
        % for each dimension
        % PIV_range ---->  x = (0.74,10.44) y=(-2.84,2.83)
        % DNS_range ---->  x = (-2.5,15.04) y=(-1.95,1.95)
        PIV_range = [0.74 10.44 ; -2.84 2.83 ] ;
        DNS_range = [-2.5 15.04 ; -1.95 1.95 ] ;
        PIV_range = PIV_range + (-3) * ...
            [ param.dX(1) -param.dX(1) ; param.dX(2) -param.dX(2)] ;
        %         DNS_range = DNS_range + (-3) * ...
        %             [ param.dX(1) -param.dX(1) ; param.dX(2) -param.dX(2)] ;
        
        filename = [param.folder_data ...
            '\XP_Irstea\wake_Re300_export_190709_4103\' ...
            'wake_Re300_export_190709_4103\B0001.dat'];
        
        nb_files_error = 2000;
        error_estim = 6e-2;
    otherwise
        error('unknown parameters');
end

% param = fct_name_file_diffusion_mode_PIV(param);
param.folder_file_U_fake_PIV = ...
    [ param.folder_file_U_temp(1:end-1) '_fake_PIV/'];
mkdir(param.folder_file_U_fake_PIV);

param_temp = param;
param_temp.a_time_dependant = true;
% param_temp = fct_name_file_diffusion_mode(param_temp);

% if (exist(param.name_file_diffusion_mode_PIV,'file')==2) || ...
%         (exist(param_temp.name_file_diffusion_mode_PIV,'file')==2)
%     if ~ (exist(param.name_file_diffusion_mode_PIV,'file')==2)
%         load(param_temp.name_file_diffusion_mode_PIV,'z_on_tau');
%         z_on_tau(:,1:param.nb_modes,:,:) = [];
%         save(param.name_file_diffusion_mode_PIV,'z_on_tau');
%     end
% else
clear param_temp

% model for the variance tensor a
a_time_dependant=param.a_time_dependant;
if a_time_dependant
    type_filter_a=param.type_filter_a;
end
M=param.M;
N_tot=param.N_tot;
d=param.d;
dt=param.dt;
lambda=param.lambda; % Energy of Chronos


%% Application of H_PIV
%% Initialization

%% Grid
% DNS grid
x_DNS = DNS_range(1,1) + param.dX(1) * (0:(param.MX(1)-1));
y_DNS = DNS_range(2,1) + param.dX(2) * (0:(param.MX(2)-1));
%     x_DNS = DNS_range(1,1):param.dX(1):DNS_range(1,2);
%     y_DNS = DNS_range(2,1):param.dX(2):DNS_range(2,2);
%     y_DNS = y_DNS - mean(y_DNS);
%     y_DNS = y_DNS - y_DNS(50);
switch param.type_data
    case {'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks',...
            'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_truncated',...
            'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_test_basis'}
        y_DNS(1:2)=[]; % Correct offset
end
y_DNS = y_DNS - mean(y_DNS); % Correct offset
[X_DNS,Y_DNS]=ndgrid(x_DNS,y_DNS);
X_DNS = X_DNS(:);Y_DNS = Y_DNS(:);


% PIV grid
filename = [param.folder_data ...
    '\XP_Irstea\wake_Re300_export_190709_4103\' ...
    'wake_Re300_export_190709_4103\B0001.dat'];
delimiter = ' ';
startRow = 5;
formatSpec = '%f%f%f%f%f%*s%*s%[^\n\r]';
fileID = fopen(filename,'r');
dataArray = textscan(fileID, formatSpec, 'Delimiter', ...
    delimiter, 'TextType', 'string', 'EmptyValue', NaN, ...
    'HeaderLines' ,startRow-1, 'ReturnOnError', false, 'EndOfLine', '\r\n');
fclose(fileID);
x_PIV_without_crop = (dataArray{1} - X_cyl(1))/cil_diameter;
y_PIV_without_crop = (dataArray{2} - X_cyl(2))/cil_diameter;


% Cropped PIV grid
% Offset due to the filtering
PIV_range = PIV_range + std_space * ...
    [ 1 -1 ; 1 -1] ;
DNS_range = DNS_range + std_space * ...
    [ 1 -1 ; 1 -1] ;
%     PIV_range = PIV_range + number_of_points_correlated * ...
%         [ param.dX(1) -param.dX(1) ; param.dX(2) -param.dX(2)] ;
%     DNS_range = DNS_range + number_of_points_correlated * ...
%         [ param.dX(1) -param.dX(1) ; param.dX(2) -param.dX(2)] ;
mask = ...
    ( x_PIV_without_crop >= max( [ PIV_range(1,1) DNS_range(1,1) ])) ...
    & ( x_PIV_without_crop <= min( [ PIV_range(1,2) DNS_range(1,2) ])) ...
    & ( y_PIV_without_crop >= max( [ PIV_range(2,1) DNS_range(2,1) ])) ...
    & ( y_PIV_without_crop <= min( [ PIV_range(2,2) DNS_range(2,2) ])) ;
x_PIV_after_crop = x_PIV_without_crop(mask);
y_PIV_after_crop = y_PIV_without_crop(mask);
MX_PIV = [ length(unique(x_PIV_after_crop)) ...
    length(unique(y_PIV_after_crop))];
M_PIV = prod(MX_PIV);

%% PIV spatial filter
switch param.type_data
    case {'DNS100_inc3d_2D_2018_11_16_blocks',...
            'DNS100_inc3d_2D_2018_11_16_blocks_truncated',...
            'DNS100_inc3d_2D_2018_11_16_blocks_test_basis'}
        x_unique_PIV = unique(x_PIV_after_crop);
        y_unique_PIV = unique(y_PIV_after_crop);
        dX_PIV = [ x_unique_PIV(2)-x_unique_PIV(1) ...
            y_unique_PIV(2)-y_unique_PIV(1) ];
        new_distance = sqrt(prod(dX_PIV));
end
number_of_points_correlated = floor(std_space/(new_distance));
dist = abs( new_distance*...
    (-number_of_points_correlated:number_of_points_correlated) );
h = exp(-(dist.^2)/(2*std_space^2));
sum_h = sum(h);
h = h/sum_h;
switch param.type_data
    case {'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks',...
            'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_truncated',...
            'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_test_basis'}
        hz = permute(h,[1 3 2]);
        z_keep = z_slice + ...
            (-number_of_points_correlated:number_of_points_correlated);
end

%% PIV error amplitude estimation
% filename_std_error = [param.folder_data ...
%     '\XP_Irstea_wake_Re300_uncertainties-2000\temporal_mean.dat'];
% if (exist(filename_std_error,'file')==2)
%     load(filename_std_error,'std_error_after_crop');
% else
%     std_error_after_crop = zeros([M_PIV,1]);
%     for idx_file = 1:nb_files_error
%         idx_file
%         filename = [param.folder_data ...
%             '\XP_Irstea_wake_Re300_uncertainties-2000\' ...
%             'wake_Re300_uncertainties-2000\B0' ...
%             num2str(idx_file,'%04.f') '.dat'];
%         %     'wake_Re300_uncertainties-2000\B00001.dat'];
%         delimiter = ' ';
%         startRow = 5;
%         formatSpec = '%*q%*q%*q%*q%f%*s%*s%*s%*s%*s%[^\n\r]';
%         % formatSpec = '%f%f%f%f%f%*s%*s%[^\n\r]';
%         fileID = fopen(filename,'r');
%         dataArray = textscan(fileID, formatSpec, 'Delimiter', ...
%             delimiter, 'TextType', 'string', 'EmptyValue', NaN, ...
%             'HeaderLines' ,startRow-1, 'ReturnOnError', false, 'EndOfLine', '\r\n');
%         fclose(fileID);
%         std_error = dataArray{1}/u_inf_measured;
%         
%         std_error_after_crop = std_error_after_crop + std_error(mask);
%         clear std_error
%     end
%     std_error_after_crop = std_error_after_crop / nb_files_error ;
%     std_error_after_crop = reshape(std_error_after_crop, MX_PIV);
%     
%     save(filename_std_error,'std_error_after_crop',...
%         'mask','x_PIV_after_crop','y_PIV_after_crop','MX_PIV',...
%         '-v7.3');
% end
% 
% 
% 
% % filename = [param.folder_data ...
% %     '\XP_Irstea_wake_Re300_uncertainties-2000\' ...
% %     'wake_Re300_uncertainties-2000\B01575.dat'];
% % delimiter = ' ';
% % startRow = 5;
% % formatSpec = '%*q%*q%*q%*q%f%*s%*s%*s%*s%*s%[^\n\r]';
% % % formatSpec = '%f%f%f%f%f%*s%*s%[^\n\r]';
% % fileID = fopen(filename,'r');
% % dataArray = textscan(fileID, formatSpec, 'Delimiter', ...
% %     delimiter, 'TextType', 'string', 'EmptyValue', NaN, ...
% %     'HeaderLines' ,startRow-1, 'ReturnOnError', false, 'EndOfLine', '\r\n');
% % fclose(fileID);
% % std_error2 = dataArray{1}/u_inf_measured;
% % 
% % std_error_after_crop2 = std_error2(mask);
% % std_error_after_crop2 = reshape(std_error_after_crop2, MX_PIV);
% % 
% % mean(std_error_after_crop(:))
% % std(std_error_after_crop(:))/mean(std_error_after_crop(:))
% % mean(std_error_after_crop2(:))
% % 
% % difft = mean(abs(std_error_after_crop(:)-std_error_after_crop2(:))) ...
% %     / mean(abs(std_error_after_crop(:)))

% mean(std_error_after_crop(:))
% figure;imagesc(unique(x_PIV_after_crop),unique(y_PIV_after_crop),...
%     std_error_after_crop');
% axis xy; axis equal;colorbar;
% keyboard;

std_error_after_crop = error_estim * ones([M_PIV 1]);
% std_error_after_crop = error_estim * ones(MX_PIV);

%%
len_blocks=param.data_in_blocks.len_blocks;
param_temp = read_data_blocks(...
    [ param.type_data(1:(end-10)) '_test_basis' ],...
    param.folder_data);
N_tot= param_temp.data_in_blocks.nb_blocks ...
    * param_temp.data_in_blocks.len_blocks;
clear param_temp
% N_tot= floor(N_tot/n_subsampl);

% Initialization
t_local=1; % index of the snapshot in a file
t_subsample=1;
big_T = param.data_in_blocks.nb_blocks; % index of the file

% t_local=1; % index of the snapshot in a file
% if param.data_in_blocks.bool % if data are saved in several files
%     big_T = 1; % index of the file
%     name_file_U_temp=param.name_file_U_temp{big_T}; % Name of the next file
% else
%     name_file_U_temp=param.name_file_U_temp; % Name of the next file
% end

param.name_file_U_fake_PIV = {};

% initialization of the index of the snapshot in the file
t_local=1;
time = 0;
interval_time = dt*(0:(len_blocks-1));
% Incrementation of the file index
big_T = big_T+1;
% Name of the new file
param.type_data=[param.data_in_blocks.type_whole_data num2str(big_T)];
% Load new file
[U,param_temp]=read_data(param.type_data,param.folder_data, ...
    param.data_in_blocks.type_whole_data,param.modified_Re);
dt = param_temp.dt;clear param_temp

% Load
% load(name_file_U_temp);
N_local = size(U,2);
U = reshape(U, [param.MX, N_local,d]);
% U = U(:,:,z_keep,:,1:d_PIV);
% U(:,1:2,:,:,:)=[];
% MX_modif = [ (param.MX(1:d_PIV) - [0 2]) ...
%     2*number_of_points_correlated+1];
switch param.data_in_blocks.type_whole_data
    case {'DNS100_inc3d_2D_2018_11_16_blocks',...
            'DNS100_inc3d_2D_2018_11_16_blocks_truncated',...
            'DNS100_inc3d_2D_2018_11_16_blocks_test_basis'}
        MX_modif = param.MX;
    case {'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks',...
            'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_truncated',...
            'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_test_basis'}
        U = U(:,:,z_keep,:,1:d_PIV);
        U(:,1:2,:,:,:)=[];
        MX_modif = [ (param.MX(1:d_PIV) - [0 2]) ...
            2*number_of_points_correlated+1];
end
U_PIV = nan([M_PIV N_local d_PIV]);
interval_time_local = time + dt*(0:(N_local-1));

%     z_on_tau=zeros(M,nb_modes_z,d,d);
%     big_T_max = 1; %BETA PARAMETER
big_T_max = size(param.name_file_U_temp,2); %BETA PARAMETER
%% Loop on time for application of H_PIV

for t=1:N_tot % loop for all time
    % for t=1:n_subsampl*N_tot % loop for all time
    time = time+dt;
    if (t_local == len_blocks + 1) % A new file needs to be loaded
        
        % for t=1:N_tot
        %     if t_local > size(U,d+1)
        clear U
        
        % Add noise
        U_PIV = U_PIV + ...
            bsxfun(@times, std_error_after_crop , randn([M_PIV N_local d_PIV]));

        % Save
        % Name of the current PIV file
        name_file_U_fake_PIV=[param.folder_file_U_fake_PIV 'strat' ...
            num2str(big_T) '_U_temp'];
        param.name_file_U_fake_PIV{big_T - param.data_in_blocks.nb_blocks} = ...
            [ name_file_U_fake_PIV '_PIV'];
        % Save current PIV file
        %             U = reshape(U_PIV, [M_PIV, N_local,d_PIV]); clear U_PIV
        U = U_PIV; clear U_PIV
        save(param.name_file_U_fake_PIV{big_T - param.data_in_blocks.nb_blocks},...
            'U',...
            'mask','x_PIV_after_crop','y_PIV_after_crop','MX_PIV',...
            'interval_time_local','dt',...
            '-v7.3');
        
        % initialization of the index of the snapshot in the file
        t_local=1;
        % Incrementation of the file index
        big_T = big_T+1;
        % Name of the new file
        param.type_data=[param.data_in_blocks.type_whole_data num2str(big_T)];
        % Load new file
        U=read_data(param.type_data,param.folder_data, ...
            param.data_in_blocks.type_whole_data,param.modified_Re);
%         if big_T > big_T_max %BETA IF CLAUSE
%             break
%         end
        
        % Load
        % Name of the new file
%         name_file_U_temp=param.name_file_U_temp{big_T};
%         % Load new file
%         load(name_file_U_temp);
        N_local = size(U,2);
        U = reshape(U, [param.MX, N_local,d]);
        switch param.data_in_blocks.type_whole_data
            case {'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks',...
                    'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_truncated',...
                    'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_test_basis'}
                U = U(:,:,z_keep,:,1:d_PIV);
                U(:,1:2,:,:,:)=[];
        end
%         U = U(:,:,z_keep,:,1:d_PIV);
%         U(:,1:2,:,:,:)=[];
        U_PIV = nan([M_PIV N_local d_PIV]);
        interval_time_local = time + dt*(0:(N_local-1));
    end
    %% Application of H_PIV
    
    % Take the current snpshot
    %     U_temp = U(:,:,:,t_local,:);
    switch param.data_in_blocks.type_whole_data
        case {'DNS100_inc3d_2D_2018_11_16_blocks',...
                'DNS100_inc3d_2D_2018_11_16_blocks_truncated',...
                'DNS100_inc3d_2D_2018_11_16_blocks_test_basis'}
            U_temp = U(:,:,t_local,:);
        case {'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks',...
                'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_truncated',...
                'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_test_basis'}
            U_temp = U(:,:,:,t_local,:);
    end
    
    switch param.data_in_blocks.type_whole_data
        case {'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks',...
                'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_truncated',...
                'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_test_basis'}
            % Spatial smoothing along z
            U_temp = sum( bsxfun(@times, U_temp, hz) ,3) ;
            U_temp = permute( U_temp ,[ 1 2 4 5 3]); % Mx x My x 1 x d_PIV
    end
%     % Spatial smoothing along z
%     U_temp = sum( bsxfun(@times, U_temp, hz) ,3) ;
%     U_temp = permute( U_temp ,[ 1 2 4 5 3]); % Mx x My x 1 x d_PIV
    
    % Spatial smoothing along x
    for k=1:d_PIV
        for j = 1:MX_modif(2)
            U_temp(:,j,1,k) = conv(U_temp(:,j,1,k)', h,'same')' ;
        end
    end
    
    % Spatial smoothing along y
    for k=1:d_PIV
        for j = 1:MX_modif(1)
            U_temp(j,:,1,k) = conv(U_temp(j,:,1,k), h,'same') ;
        end
    end
    
    %       conv(u,v,'same')
    %       conv(u,v,'valid')
    
    % Interpolation
    U_PIV_temp = nan([M_PIV 1 d_PIV]);
    for k=1:d_PIV
        U_PIV_temp(:,1,k) = ...
            interp2(x_DNS, y_DNS, U_temp(:,:,1,k)', ...
            x_PIV_after_crop, y_PIV_after_crop, ...
            'linear');
    end
    clear U_temp
    
    % Concatenate PIV snapshots
    U_PIV(:,t_local,:) = U_PIV_temp; clear U_PIV_temp
    
    t_local=t_local+1;
end
clear U

% Add noise
U_PIV = U_PIV + ...
    bsxfun(@times, std_error_after_crop , randn([M_PIV N_local d_PIV]));

% Save
% Name of the current PIV file
name_file_U_fake_PIV=[param.folder_file_U_fake_PIV 'strat' ...
            num2str(big_T) '_U_temp'];
param.name_file_U_fake_PIV{big_T - param.data_in_blocks.nb_blocks} = ...
    [ name_file_U_fake_PIV '_PIV'];
% Save current PIV file
%     U = reshape(U_PIV, [M_PIV, N_local,d_PIV]); clear U_PIV
U = U_PIV; clear U_PIV
save(param.name_file_U_fake_PIV{big_T - param.data_in_blocks.nb_blocks},...
    'U',...
    'mask','x_PIV_after_crop','y_PIV_after_crop','MX_PIV',...
    'interval_time_local','dt',...
    '-v7.3');
