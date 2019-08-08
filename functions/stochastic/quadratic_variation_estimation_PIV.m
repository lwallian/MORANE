function [param,z_on_tau]=quadratic_variation_estimation_PIV(param,bt)
% Variance tensor estimation
%

%% Get parameters

switch param.type_data
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
        
    otherwise
        error('unknown parameters');
end

param = fct_name_file_diffusion_mode_PIV(param);
param.folder_file_U_temp_PIV = ...
    [ param.folder_file_U_temp(1:end-1) '_PIV/'];
mkdir(param.folder_file_U_temp_PIV);

param_temp = param;
param_temp.a_time_dependant = true;
param_temp = fct_name_file_diffusion_mode(param_temp);

if (exist(param.name_file_diffusion_mode_PIV,'file')==2) || ...
        (exist(param_temp.name_file_diffusion_mode_PIV,'file')==2)
    if ~ (exist(param.name_file_diffusion_mode_PIV,'file')==2)
        load(param_temp.name_file_diffusion_mode_PIV,'z_on_tau');
        z_on_tau(:,1:param.nb_modes,:,:) = [];
        save(param.name_file_diffusion_mode_PIV,'z_on_tau');
    end
else
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
    
    % PIV spatial filter
    number_of_points_correlated = floor(std_space/(new_distance));
    dist = abs( new_distance*...
        (-number_of_points_correlated:number_of_points_correlated) );
    h = exp(-(dist.^2)/(2*std_space^2));
    sum_h = sum(h);
    h = h/sum_h;
%     hx = permute(h,[2 1]);
%     hy = h ;
    hz = permute(h,[1 3 2]);
%     clear h;
    z_keep = z_slice + ...
        (-number_of_points_correlated:number_of_points_correlated);
    
    % DNS grid
    x_DNS = DNS_range(1,1) + param.dX(1) * (0:(param.MX(1)-1));
    y_DNS = DNS_range(2,1) + param.dX(2) * (0:(param.MX(2)-1));
%     x_DNS = DNS_range(1,1):param.dX(1):DNS_range(1,2);
%     y_DNS = DNS_range(2,1):param.dX(2):DNS_range(2,2);
%     y_DNS = y_DNS - mean(y_DNS);
%     y_DNS = y_DNS - y_DNS(50); 
    y_DNS(1:2)=[]; % Correct offset
    y_DNS = y_DNS - mean(y_DNS); % Correct offset
    [X_DNS,Y_DNS]=ndgrid(x_DNS,y_DNS);
    X_DNS = X_DNS(:);Y_DNS = Y_DNS(:);
    
    % PIV grid    
    filename = [param.folder_data ...
        '\DATA_XP_Irstea\wake_Re300_export_190709_4103\' ...
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
    
    t_local=1; % index of the snapshot in a file
    if param.data_in_blocks.bool % if data are saved in several files
        big_T = 1; % index of the file
        name_file_U_temp=param.name_file_U_temp{big_T}; % Name of the next file
    else
        name_file_U_temp=param.name_file_U_temp; % Name of the next file
    end
    param.name_file_U_temp_PIV = {};
    
    % Load
    load(name_file_U_temp);
    N_local = size(U,2);
    U = reshape(U, [param.MX, N_local,d]);
    U = U(:,:,z_keep,:,1:d_PIV);
    U(:,1:2,:,:,:)=[];
    MX_modif = [ (param.MX(1:d_PIV) - [0 2]) ...
        2*number_of_points_correlated+1];
    U_PIV = nan([M_PIV N_local d_PIV]);
    
%     z_on_tau=zeros(M,nb_modes_z,d,d);
%     big_T_max = 1; %BETA PARAMETER
    big_T_max = size(param.name_file_U_temp,2); %BETA PARAMETER
    %% Loop on time for application of H_PIV
    for t=1:N_tot
        if t_local > size(U,d+1) 
            clear U
            % Save
            % Name of the current PIV file
            name_file_U_temp_PIV=[param.folder_file_U_temp_PIV 'strat' ...
                            num2str(big_T) '_U_temp'];
            param.name_file_U_temp_PIV{big_T} = ...
                [ name_file_U_temp_PIV '_PIV'];
            % Save current PIV file
            %             U = reshape(U_PIV, [M_PIV, N_local,d_PIV]); clear U_PIV
            U = U_PIV; clear U_PIV
            save(param.name_file_U_temp_PIV{big_T},'U',...
                'mask','x_PIV_after_crop','y_PIV_after_crop','MX_PIV',...
                '-v7.3');
                        
            % initialization of the index of the snapshot in the file
            t_local=1;
            % Incrementation of the file index
            big_T = big_T+1
            if big_T > big_T_max %BETA IF CLAUSE
                break
            end
            
            % Load
            % Name of the new file
            name_file_U_temp=param.name_file_U_temp{big_T};
            % Load new file
            load(name_file_U_temp);
            N_local = size(U,2);
            U = reshape(U, [param.MX, N_local,d]);
            U = U(:,:,z_keep,:,1:d_PIV);
            U(:,1:2,:,:,:)=[];
            U_PIV = nan([M_PIV N_local d_PIV]);
        end
        %% Application of H_PIV
        
        % Take the current snpshot
        U_temp = U(:,:,:,t_local,:);
        
        % Spatial smoothing along z
        U_temp = sum( bsxfun(@times, U_temp, hz) ,3) ;
        U_temp = permute( U_temp ,[ 1 2 4 5 3]); % Mx x My x 1 x d_PIV
        
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
%     % Save
%     % Name of the current PIV file
%     param.name_file_U_temp_PIV{big_T} = ...
%         [ name_file_U_temp_PIV '_PIV'];
%     % Save current PIV file
% %     U = reshape(U_PIV, [M_PIV, N_local,d_PIV]); clear U_PIV
%     U = U_PIV; clear U_PIV
%     save(param.name_file_U_temp_PIV{big_T},'U',...
%                 'mask','x_PIV_after_crop','y_PIV_after_crop','MX_PIV',...
%                 '-v7.3');
    
    %% Variance tensor computation
    %% Initialization
    if a_time_dependant && strcmp(type_filter_a,'b_i')
        % a(x,t) = z_0(x) + \sum b_i(t) z_i(x)
        % number of modes z_j
        nb_modes_z = param.nb_modes+1;
        % Projection basis
        % Add the constant 1 to Chronos
        bt = [ bt'; ones(1,N_tot)];% nb_modes_z x N_tot
        % Energy of the projection basis
        lambda= [lambda; 1]; % nb_modes_z
    elseif ~a_time_dependant
        % a(x,t) = z_0(x)
        % number of modes z_j
        nb_modes_z =1;
        % Projection basis
        % Use only the constant 1
        bt = ones(1,N_tot);% nb_modes_z x N_tot
        % Energy of the projection basis
        lambda=1;
    else
        error('not coded or deprecated');
    end
    
    t_local=1; % index of the snapshot in a file
    if param.data_in_blocks.bool % if data are saved in several files
        big_T = 1; % index of the file
        name_file_U_temp=param.name_file_U_temp_PIV{big_T}; % Name of the next file
    else
        name_file_U_temp=param.name_file_U_temp_PIV{1}; % Name of the next file
    end
    % Load
    load(name_file_U_temp);
    
%     [M_PIV,~,d_PIV] = size(U);
    z_on_tau=zeros(M_PIV,nb_modes_z,d_PIV,d_PIV);
%     big_T_max = size(param.name_file_U_temp,2); %BETA PARAMETER
    %% Projection
    %     if big_data
    for t=1:N_tot
        if t_local > size(U,2) % A new file needs to be loaded
            % initialization of the index of the snapshot in the file
            t_local=1;
            % Incrementation of the file index
            big_T = big_T+1
            if big_T > big_T_max %BETA IF CLAUSE
                break
            end
            % Name of the new file
            name_file_U_temp=param.name_file_U_temp_PIV{big_T};
            % Load new file
            load(name_file_U_temp);
        end
        % Projection
        for k_mode=1:nb_modes_z
            for i=1:d_PIV
                for j=1:d_PIV
                    z_on_tau(:,k_mode,i,j)=z_on_tau(:,k_mode,i,j) ...
                        + bt(k_mode,t)/lambda(k_mode) ...
                        * U(:,t_local,i) .* U(:,t_local,j);% M x 1 x d x d
                end
            end
        end
        % Incrementation of the index of the snapshot in the file
        t_local=t_local+1;
    end
    clear U;
    
    z_on_tau=1/N_tot*z_on_tau;
    
%     % Normalization and time step influence
%     z_on_tau=dt/N_tot*z_on_tau;
%     
%     % To circumvent the effect of the threshold of the downsampling rate
%     if strcmp(param.decor_by_subsampl.choice_n_subsample, 'corr_time')
%         z_on_tau = z_on_tau * param.decor_by_subsampl.tau_corr / param.decor_by_subsampl.n_subsampl_decor;
%     end
    
    %% Save
%     if nargout < 2
        save(param.name_file_diffusion_mode_PIV,'z_on_tau',...
                'mask','x_PIV_after_crop','y_PIV_after_crop','MX_PIV',...
                '-v7.3');
        %     save(name_file_mode,'z_on_tau','-append');
%         clear z_on_tau
%     end
    
    %% Inversion of HSigSigH    
    
    z_on_tau(:,:,1,1) = z_on_tau(:,:,1,1) + (0.06)^2 ;
    z_on_tau(:,:,2,2) = z_on_tau(:,:,2,2) + (0.06)^2 ;
    
    inv_HSigSigH = nan(size(z_on_tau));
    det_HSigSigH = z_on_tau(:,:,1,1) .* z_on_tau(:,:,2,2) ...
        - z_on_tau(:,:,1,2) .* z_on_tau(:,:,2,1) ;
    inv_HSigSigH(:,:,1,1) = z_on_tau(:,:,2,2);
    inv_HSigSigH(:,:,2,2) = z_on_tau(:,:,1,1); 
    inv_HSigSigH(:,:,1,2) = - z_on_tau(:,:,1,2); 
    inv_HSigSigH(:,:,2,1) = - z_on_tau(:,:,2,1); 
    inv_HSigSigH = bsxfun(@times, 1./det_HSigSigH , inv_HSigSigH );
    
    %% Save
%     if nargout < 2
        save(param.name_file_HSigSigH_PIV,'inv_HSigSigH',...
                'mask','x_PIV_after_crop','y_PIV_after_crop','MX_PIV',...
                '-v7.3');
        %     save(name_file_mode,'z_on_tau','-append');
%     end
end