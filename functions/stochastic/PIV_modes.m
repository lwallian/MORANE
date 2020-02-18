function PIV_modes(param)
% Variance tensor estimation
%

%% Get parameters

switch param.type_data
    case {'DNS100_inc3d_2D_2018_11_16_blocks',...
            'DNS100_inc3d_2D_2018_11_16_blocks_truncated',...
            'DNS100_inc3d_2D_2018_11_16_blocks_test_basis'}
        
        % Smoothing
                new_distance = 0.041666666666666664;
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
        DNS_range_ref = [-2.5 15.04 ; -1.95 1.95 ] ;
        PIV_range = PIV_range + (-3) * ...
            [ 0.0417 -0.0417 ; 0.0417 -0.0417] ;
%             [ param.dX(1) -param.dX(1) ; param.dX(2) -param.dX(2)] ;
        %         DNS_range = [-2.5 15.04 ; -1.95 1.95 ] ;
%         PIV_range = PIV_range + (-3) * ...
%             [ param.dX(1) -param.dX(1) ; param.dX(2) -param.dX(2)] ;
        %         DNS_range = DNS_range + (-3) * ...
        %             [ param.dX(1) -param.dX(1) ; param.dX(2) -param.dX(2)] ;
        
%         filename = [param.folder_data '\XP_Irstea' ...
%             '\wake_Re100_export_190710_4107\' ...
        filename = [param.folder_data(1:end-1) '_PIV\'  ...
            '\wake_Re100\' ...
            'B0001.dat'];
        warning('The grid of Re 300 PIV measurement is used');
        
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
        DNS_range_ref = [-2.5 15.04 ; -1.95 1.95 ] ;
        PIV_range = PIV_range + (-3) * ...
            [ param.dX(1) -param.dX(1) ; param.dX(2) -param.dX(2)] ;
        %         DNS_range = DNS_range + (-3) * ...
        %             [ param.dX(1) -param.dX(1) ; param.dX(2) -param.dX(2)] ;
        
%         filename = [param.folder_data ...
%             '\XP_Irstea\wake_Re300_export_190709_4103\' ...
%             'wake_Re300_export_190709_4103\B0001.dat'];
        filename = [param.folder_data(1:end-1) '_PIV\' ...
            'wake_Re300\B0001.dat'];
        
        nb_files_error = 2000;
        error_estim = 6e-2;
    otherwise
        error('unknown parameters');
end

%% Load
load(param.name_file_mode,'phi_m_U')

param.name_file_mode_PIV = [ param.name_file_mode(1:end-4) '_PIV.mat'];
param.name_file_mode_PIV = replace(param.name_file_mode_PIV,'data','data_PIV');
% mkdir(param.name_file_mode_PIV);

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
% clear param_temp
% 
% % model for the variance tensor a
% a_time_dependant=param.a_time_dependant;
% if a_time_dependant
%     type_filter_a=param.type_filter_a;
% end
% M=param.M;
% N_tot=param.N_tot;
d=param.d;
% dt=param.dt;
% lambda=param.lambda; % Energy of Chronos


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
%         filename = [param.folder_data ...
%             '\XP_Irstea\wake_Re300_export_190709_4103\' ...
%             'wake_Re300_export_190709_4103\B0001.dat'];
filename = [param.folder_data(1:end-1) '_PIV\' ...
    'wake_Re300\B0001.dat'];
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
DNS_range_ref = DNS_range_ref + std_space * ...
    [ 1 -1 ; 1 -1] ;
%     PIV_range = PIV_range + number_of_points_correlated * ...
%         [ param.dX(1) -param.dX(1) ; param.dX(2) -param.dX(2)] ;
%     DNS_range = DNS_range + number_of_points_correlated * ...
%         [ param.dX(1) -param.dX(1) ; param.dX(2) -param.dX(2)] ;
mask = ...
    ( x_PIV_without_crop >= max( [ PIV_range(1,1) DNS_range_ref(1,1) ])) ...
    & ( x_PIV_without_crop <= min( [ PIV_range(1,2) DNS_range_ref(1,2) ])) ...
    & ( y_PIV_without_crop >= max( [ PIV_range(2,1) DNS_range_ref(2,1) ])) ...
    & ( y_PIV_without_crop <= min( [ PIV_range(2,2) DNS_range_ref(2,2) ])) ;
x_PIV_after_crop = x_PIV_without_crop(mask);
y_PIV_after_crop = y_PIV_without_crop(mask);
MX_PIV = [ length(unique(x_PIV_after_crop)) ...
    length(unique(y_PIV_after_crop))];
M_PIV = prod(MX_PIV);

%% PIV spatial filter
x_unique_PIV = unique(x_PIV_after_crop);
y_unique_PIV = unique(y_PIV_after_crop);
dX_PIV = [ x_unique_PIV(2)-x_unique_PIV(1) ...
    y_unique_PIV(2)-y_unique_PIV(1) ];
% switch param.type_data
%     case {'DNS100_inc3d_2D_2018_11_16_blocks',...
%             'DNS100_inc3d_2D_2018_11_16_blocks_truncated',...
%             'DNS100_inc3d_2D_2018_11_16_blocks_test_basis'}
% %         x_unique_PIV = unique(x_PIV_after_crop);
% %         y_unique_PIV = unique(y_PIV_after_crop);
% %         dX_PIV = [ x_unique_PIV(2)-x_unique_PIV(1) ...
% %             y_unique_PIV(2)-y_unique_PIV(1) ];
%         new_distance = sqrt(prod(dX_PIV));
% end
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

phi_m_U = reshape(phi_m_U, [param.MX, param.nb_modes+1,d]);

switch param.data_in_blocks.type_whole_data
%     case {'DNS100_inc3d_2D_2018_11_16_blocks',...
%             'DNS100_inc3d_2D_2018_11_16_blocks_truncated',...
%             'DNS100_inc3d_2D_2018_11_16_blocks_test_basis'}
%         MX_modif = param.MX;
    case {'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks',...
            'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_truncated',...
            'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_test_basis'}
        phi_m_U = phi_m_U(:,:,z_keep,:,1:d_PIV);
        phi_m_U(:,1:2,:,:,:)=[];
end
MX_modif = [ (param.MX(1:d_PIV) - [0 2]) ...
    2*number_of_points_correlated+1];
phi_m_U_PIV = nan([M_PIV param.nb_modes+1 d_PIV]);

%% Loop on time for application of H_PIV
for idx_modes=1:param.nb_modes+1 % loop for all time
    %% Application of H_PIV
    % Take the current snpshot
    switch param.data_in_blocks.type_whole_data
        case {'DNS100_inc3d_2D_2018_11_16_blocks',...
                'DNS100_inc3d_2D_2018_11_16_blocks_truncated',...
                'DNS100_inc3d_2D_2018_11_16_blocks_test_basis'}
            phi_m_U_temp = phi_m_U(:,:,idx_modes,:);
        case {'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks',...
                'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_truncated',...
                'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_test_basis'}
            phi_m_U_temp = phi_m_U(:,:,:,idx_modes,:);
    end
    
    switch param.data_in_blocks.type_whole_data
        case {'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks',...
                'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_truncated',...
                'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_test_basis'}
            % Spatial smoothing along z
            phi_m_U_temp = sum( bsxfun(@times, phi_m_U_temp, hz) ,3) ;
            phi_m_U_temp = permute( phi_m_U_temp ,[ 1 2 4 5 3]); % Mx x My x 1 x d_PIV
    end
%     % Spatial smoothing along z
%     phi_m_U_temp = sum( bsxfun(@times, phi_m_U_temp, hz) ,3) ;
%     phi_m_U_temp = permute( phi_m_U_temp ,[ 1 2 4 5 3]); % Mx x My x 1 x d_PIV
    
    % Spatial smoothing along x
    for k=1:d_PIV
        for j = 1:MX_modif(2)
            phi_m_U_temp(:,j,1,k) = conv(phi_m_U_temp(:,j,1,k)', h,'same')' ;
        end
    end
    
    % Spatial smoothing along y
    for k=1:d_PIV
        for j = 1:MX_modif(1)
            phi_m_U_temp(j,:,1,k) = conv(phi_m_U_temp(j,:,1,k), h,'same') ;
        end
    end
    
    % Interpolation
    phi_m_U_PIV_temp = nan([M_PIV 1 d_PIV]);
    for k=1:d_PIV
        phi_m_U_PIV_temp(:,1,k) = ...
            interp2(x_DNS, y_DNS, phi_m_U_temp(:,:,1,k)', ...
            x_PIV_after_crop, y_PIV_after_crop, ...
            'linear');
    end
    clear phi_m_U_temp
    
    % Concatenate PIV snapshots
    phi_m_U_PIV(:,idx_modes,:) = phi_m_U_PIV_temp; clear phi_m_U_PIV_temp
end
clear phi_m_U

% Save
phi_m_U = phi_m_U_PIV; clear phi_m_U_PIV
save(param.name_file_mode_PIV,...
    'phi_m_U',...
    'mask','x_PIV_after_crop','y_PIV_after_crop','MX_PIV',...
    '-v7.3');
