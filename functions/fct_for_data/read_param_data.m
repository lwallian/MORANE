function param=read_param_data(type_data,folder_data,type_whole_data,modified_Re)
% Load data
%

folder_data_ref=folder_data;
if nargin < 4
    modified_Re = false;
end

% For 'incompact3d_wake_episode3_cut_truncated'
% nb_periods = 15; % nb of periods in the learning set
nb_periods = 10; % nb of periods in the learning set
T_period = 4.1; % Time perido of the system
subsample=4;
% subsample=1;
t_begin = 20;
        
%% Load data

switch type_data
    case 'inc3D_HRLESlong_Re3900_blocks1'
        load([ folder_data 'file_HRLES3900_1.mat']);
        
        [Mx, My, Mz, N_tot,d] = size(U);
        N_test=N_tot-1;
        MX=[Mx My Mz];
        M=prod(MX);
        a=dX(1)*(1:MX(1))';
        b=dX(2)*(1:MX(2))';
        c=dX(3)*(1:MX(3))';
        grid = {a b c};
        
        clear U
        
    case 'incompact3d_wake_episode3_cut'
        load([ folder_data 'data_incompact3d_wake_episode3_cut.mat']);
        
        %         U(:,:,2001:end,:)=[];
        
        [Mx,My,N_tot,d]=size(U);
        N_test=N_tot-1;
        MX=[Mx My];
        M=prod(MX);
        dx=dX(1);
        dy=dX(2);
        a = dx*(1:Mx)';
        b = dy*(1:My);
        grid = {a b'};
        
        clear U
        
    case 'incompact3d_wake_episode3_cut_truncated'
        load([ folder_data 'data_incompact3d_wake_episode3_cut.mat']);
%         
%         nb_periods = 10;
%         T_period = 4.1;
        
        warning('the first quarter of data is removed because of the intermittency');
        n_begin = ceil(t_begin /dt);
        U(:,:,1:max([1 (n_begin-1)]),:)=[];

        warning('the last quarter of data is removed and keep for the tests');
        t_periods = T_period*nb_periods;
        N_dt_periods = ceil(t_periods/dt);
        U(:,:,N_dt_periods:end,:)=[];
        
        U = U(:,:,1:subsample:end,:) ;
        dt = dt * subsample;
        
        [Mx,My,N_tot,d]=size(U);
        N_test=N_tot-1;
        MX=[Mx My];
        M=prod(MX);
        dx=dX(1);
        dy=dX(2);
        a = dx*(1:Mx)';
        b = dy*(1:My);
        grid = {a b'};
        
        clear U
        
    case 'incompact3d_wake_episode3_cut_test_basis'
        load([ folder_data 'data_incompact3d_wake_episode3_cut.mat']);
        
%         nb_periods = 10;
%         T_period = 4.1;
        
        n_begin = ceil(t_begin /dt);
        U(:,:,1:max([1 (n_begin-1)]),:)=[];
        
        t_periods = T_period*nb_periods;
        N_dt_periods = ceil(t_periods/dt);
        U(:,:,1:(N_dt_periods-1),:)=[];
        
        U = U(:,:,1:subsample:end,:) ;
        dt = dt * subsample;
        
        [Mx,My,N_tot,d]=size(U);
        N_test=N_tot-1;
        MX=[Mx My];
        M=prod(MX);
        dx=dX(1);
        dy=dX(2);
        a = dx*(1:Mx)';
        b = dy*(1:My);
        grid = {a b'};
        
        clear U
    
    case 'incompact3D_noisy2D'
        load([ folder_data 'data_incompact3d_wake_noisy_1.mat']);
        clear P Vort
        dX(3)=[];
        
        N_test = 800;
        
        [Mx,My,N_tot,d]=size(U);
        MX=[Mx My];
        M=prod(MX);
        dx=dX(1);
        dy=dX(2);
        a = dx*(1:Mx)';
        b = dy*(1:My);
        grid = {a b'};
        
        if sub_sampling
            i=1;
            Unew=nan([ MX N_tot/n_sub_sampling d] );
            for k=1:N_tot
                if mod(k,n_sub_sampling)==0
                    Unew(:,:,i,:)=U(:,:,k,:);
                    i=i+1;
                end
            end
            U=Unew; clear Unew i
            N_tot = N_tot/n_sub_sampling;
            dt=dt*n_sub_sampling;
            
            N_test = ceil(N_test/n_sub_sampling);
            save([ folder_data 'data_incompact3d_wake_noisy_1_sub_sampl'], ...
                'U','dt','Re','dX','normalized');
        end
        
        clear U
        
    case 'incompact3D_noisy2D_10dt'
        load([ folder_data 'data_incompact3d_wake_noisy_1_10dt.mat']);
        
        U(:,:,2001:end,:)=[];
        N_test = 1500;
        %         % N =3000;
        %         N_test = 2000;
        
        [Mx,My,N_tot,d]=size(U);
        MX=[Mx My];
        M=prod(MX);
        dx=dX(1);
        dy=dX(2);
        a = dx*(1:Mx)';
        b = dy*(1:My);
        grid = {a b'};
        
        clear U
        
    case 'incompact3D_noisy2D_10dt_tronc'
        load([ folder_data 'data_incompact3d_wake_noisy_1_10dt_tronc.mat']);
        % Need ~ 6 Go of RAM
        
        N_test = 1000;
        
        [Mx,My,N_tot,d]=size(U);
        MX=[Mx My];
        M=prod(MX);
        dx=dX(1);
        dy=dX(2);
        a = dx*(1:Mx)';
        b = dy*(1:My);
        grid = {a b'};
        
        clear U
        
    case 'incompact3D_noisy2D_10dt_subsampl'
        load([ folder_data 'data_incompact3d_wake_noisy_1_10dt_sub_sampl.mat']);
        % Need ~ 1 Go of RAM
        
        N_test = 1999;
        
        [Mx,My,N_tot,d]=size(U);
        MX=[Mx My];
        M=prod(MX);
        dx=dX(1);
        dy=dX(2);
        a = dx*(1:Mx)';
        b = dy*(1:My);
        grid = {a b'};
        clear U
        
        rho0=1e3;
        
    case 'incompact3D_noisy2D_20dt_subsampl'
        if exist([ folder_data 'data_incompact3d_wake_noisy_1_20dt_sub_sampl.mat'],'file')
            load([ folder_data 'data_incompact3d_wake_noisy_1_20dt_sub_sampl.mat']);
        else
            load([ folder_data 'data_incompact3d_wake_noisy_1_10dt_sub_sampl.mat']);
            
            n_sub_sampling=2;
            
            folder_data=folder_data_ref;
            
            [Mx,My,N_tot,d]=size(U);
            MX=[Mx My];
            i=1;
            Unew=nan([ MX N_tot/n_sub_sampling d] );
            for k=1:N_tot
                if mod(k,n_sub_sampling)==0
                    Unew(:,:,i,:)=U(:,:,k,:);
                    i=i+1;
                end
            end
            U=Unew; clear Unew i
            N_tot = N_tot/n_sub_sampling;
            dt=dt*n_sub_sampling;
            
            
            save([ folder_data 'data_incompact3d_wake_noisy_1_20dt_sub_sampl'], ...
                'U','dt','Re','dX','normalized');
            
        end
        
        N_test = 999;
        
        [Mx,My,N_tot,d]=size(U);
        MX=[Mx My];
        M=prod(MX);
        dx=dX(1);
        dy=dX(2);
        a = dx*(1:Mx)';
        b = dy*(1:My);
        grid = {a b'};
        clear U
        
        rho0=1e3;
        
    case 'incompact3D_noisy2D_subsampl'
        load([ folder_data 'data_incompact3d_wake_noisy_1_sub_sampl.mat']);
        clear P Vort
        dX(3)=[];
        
        N_test = 80;
        
        [Mx,My,N_tot,d]=size(U);
        MX=[Mx My];
        M=prod(MX);
        dx=dX(1);
        dy=dX(2);
        a = dx*(1:Mx)';
        b = dy*(1:My);
        grid = {a b'};
        clear U
        
    case 'incompact3D_noisy2D_subsampl_repmat3D'
        load([ folder_data 'data_incompact3d_wake_noisy_1_sub_sampl.mat']);
        clear P Vort
        dX(3)=[];
        
        N_test = 80;
        
        [Mx,My,N_tot,~]=size(U);
        
        Mz=Mx/10;
        dz=dX(1);
        d=3;
        U =permute(U,[1 2 5 3 4]);
        U = repmat(U, [ 1 1 Mz 1 1]);
        dX = [ dX dz];
        
        MX=[Mx My Mz];
        M=prod(MX);
        dx=dX(1);
        dy=dX(2);
        dz=dX(3);
        a = dx*(1:Mx)';
        b = dy*(1:My);
        c= dz*(1:Mz)';
        grid = {a b' c};
        c = permute(c, [ 3 1 2]);
        
        U(:,:,:,:,3)=zeros([MX N_tot]);
        
        idx_permute = [3 1 2];
        U = permute(U,[idx_permute 4 5]);
        U = U(:,:,:,:,idx_permute);
        MX = MX(idx_permute);
        dX = dX(idx_permute);
        grid = grid(idx_permute);
        
        clear U
        
    case 'incompact3D_noisy2D_subsampl_flou'
        load([ folder_data 'data_incompact3d_wake_noisy_1_sub_sampl_flou.mat']);
        %         clear P Vort
        %         dX(3)=[];
        
        N_test = 100;
        
        [Mx,My,N_tot,d]=size(U);
        MX=[Mx My];
        M=prod(MX);
        dx=dX(1);
        dy=dX(2);
        a = dx*(1:Mx)';
        b = dy*(1:My);
        grid = {a b'};
        clear U
        
        
    case 'dominique_config13'
        load([ folder_data 'config13.mat'])
        % N_test = 0;
        N_test = 400;
        
        grid = {a b'};
        dx=a(2)-a(1);
        dy=b(2)-b(1);
        dz=nan;
        dX=[dx dy];
        
    case 'dominique_config12'
        load([ folder_data 'config12.mat'])
        N_test = 200;
        % N_test = 80;
        
        dt=dt*2; % !!!!!!!!!!!!
        
        grid = {a b'};
        dx=a(2)-a(1);
        dy=b(2)-b(1);
        dz=nan;
        dX=[dx dy];
        
    case 'dominique_config12bis'
        load([ folder_data 'config12bis.mat'])
        N_test = 200;
        % N_test = 80;
        
        grid = {a b'};
        dx=a(2)-a(1);
        dy=b(2)-b(1);
        dz=nan;
        dX=[dx dy];
    case 'test'
        load([ folder_data 'data_test.mat'])
        
        [M,N_tot,d]=size(U);
        N_test = ceil(N_tot/3);
        
        grid = {a b'};
        dx=a(2)-a(1);
        dy=b(2)-b(1);
        dz=nan;
        dX=[dx dy];
        
    case 'test2'
        load([ folder_data 'data_test2.mat'])
        
        [M,N_tot,d]=size(U);
        N_test = ceil(N_tot/3);
        
        grid = {a b'};
        dx=a(2)-a(1);
        dy=b(2)-b(1);
        dz=nan;
        dX=[dx dy];
        V0=1;
        
    case 'test3'
        load([ folder_data 'data_test3.mat'])
        
        [M,N_tot,d]=size(U);
        N_test = ceil(N_tot/3);
        
        grid = {a b'};
        dx=a(2)-a(1);
        dy=b(2)-b(1);
        dz=nan;
        dX=[dx dy];
        V0=1;
        
    case 'test_3D'
        load([ folder_data 'data_test_3D.mat'])
        
        [M,N_tot,d]=size(U);
        N_test = ceil(N_tot/3);
        
        grid = {a b' squeeze(c)};
        dx=a(2)-a(1);
        dy=b(2)-b(1);
        dz=c(2)-c(1);
        dX=[dx dy dz];
        V0=1;
        
    case 'test_3D2'
        load([ folder_data 'data_test_3D2.mat'])
        
        [M,N_tot,d]=size(U);
        N_test = ceil(N_tot/3);
        
        grid = {a b' squeeze(c)};
        dx=a(2)-a(1);
        dy=b(2)-b(1);
        dz=c(2)-c(1);
        dX=[dx dy dz];
        V0=1;
        
    case 'adda'
        load([ folder_data 'dato_carlier3.mat'])
        
        rho0=1e3;
        normalized=true;
        nu = 1.5e-5;
        Re =3900;
        D=32e-3;
        U_cara = Re*nu/D;
        T_cara = D/U_cara;
        dt =  dt / T_cara;
        
        %N_test = 250;
        N_test = 300;
        
        grid = {a b'};
        dx=a(2)-a(1);
        dy=b(2)-b(1);
        dz=nan;
        dX=[dx dy];
        
    case 'LES_3D_1_sub_sample'
        load([ folder_data '3D_LES_sub_sample1.mat'])
        % Need ~ 2 Go of RAM
        
        N_test = 74;
        [Mx, My, Mz, N_tot,d] = size(U);
        MX=[Mx My Mz];
        M=prod(MX);
        a=dX(1)*(1:MX(1))';
        b=dX(2)*(1:MX(2))';
        c=dX(3)*(1:MX(3))';
        grid = {a b c};
        clear U
        
    case 'LES_3D_tot_sub_sample'
        load([ folder_data '3D_LES_sub_sample_total.mat'])
        
        N_test = 299;
        [Mx, My, Mz, N_tot,d] = size(U);
        MX=[Mx My Mz];
        M=prod(MX);
        a=dX(1)*(1:MX(1))';
        b=dX(2)*(1:MX(2))';
        c=dX(3)*(1:MX(3))';
        grid = {a b c};
        clear U
        
    case 'LES_3D_1'
        load([ folder_data '3D_LES_1.mat'])
        % Need ~ 30 Go of RAM
        
        N_test = 74;
        d=3;
        [Mx, My, Mz, N_tot] = size(u);
        MX=[Mx My Mz];
        M=prod(MX);
        a=dX(1)*(1:MX(1))';
        b=dX(2)*(1:MX(2))';
        c=dX(3)*(1:MX(3))';
        grid = {a b c};
        
        U=u; clear u;
        U(:,:,:,:,2)=v; clear v;
        U(:,:,:,:,3)=w; clear w;
        clear U
        
    case 'LES_3D_2'
        load([ folder_data '3D_LES_2.mat'])
        
        N_test = 74;
        d=3;
        [Mx, My, Mz, N_tot] = size(u);
        MX=[Mx My Mz];
        M=prod(MX);
        a=dX(1)*(1:MX(1))';
        b=dX(2)*(1:MX(2))';
        c=dX(3)*(1:MX(3))';
        grid = {a b c};
        
        U=u; clear u;
        U(:,:,:,:,2)=v; clear v;
        U(:,:,:,:,3)=w; clear w;
        clear U
        
    case 'LES_3D_3'
        load([ folder_data '3D_LES_3.mat'])
        
        N_test = 74;
        d=3;
        [Mx, My, Mz, N_tot] = size(u);
        MX=[Mx My Mz];
        M=prod(MX);
        a=dX(1)*(1:MX(1))';
        b=dX(2)*(1:MX(2))';
        c=dX(3)*(1:MX(3))';
        grid = {a b c};
        
        U=u; clear u;
        U(:,:,:,:,2)=v; clear v;
        U(:,:,:,:,3)=w; clear w;
        clear U
        
    case 'LES_3D_4'
        load([ folder_data '3D_LES_4.mat'])
        
        N_test = 74;
        d=3;
        [Mx, My, Mz, N_tot] = size(u);
        MX=[Mx My Mz];
        M=prod(MX);
        a=dX(1)*(1:MX(1))';
        b=dX(2)*(1:MX(2))';
        c=dX(3)*(1:MX(3))';
        grid = {a b c};
        
        U=u; clear u;
        U(:,:,:,:,2)=v; clear v;
        U(:,:,:,:,3)=w; clear w;
        clear U
        
    case 'incompact3D_noisy2D_40dt_subsampl'
        load([ folder_data 'data_incompact3d_wake_noisy_1_40dt_sub_sampl.mat']);
        
        N_test = 499;
        
        [Mx,My,N_tot,d]=size(U);
        MX=[Mx My];
        M=prod(MX);
        dx=dX(1);
        dy=dX(2);
        a = dx*(1:Mx)';
        b = dy*(1:My);
        grid = {a b'};
        clear U
        
        rho0=1e3;
        
        
    case 'incompact3D_noisy2D_40dt_subsampl_truncated'
        load([ folder_data 'incompact3D_noisy2D_40dt_subsampl_truncated_U_centered.mat'])
        
        N_test = 374;
        normalized = 1;
        Re = 300;
        
        Mx = 146;
        My = 107;
        MX = [Mx, My];
        [M,N_tot,d]=size(U);
        dX = [0.1871, 0.1869];
        dt = 0.08;
        dx = dX(1);
        dy = dX(2);
        a = dx * (1 : Mx)';
        b = dy * (1 : My)';
        grid = {a, b'};
        clear U;
        
        rho0=1e3;
        
    case 'test_1_block'
        load([ folder_data 'data_test_1_block.mat']);
        
        N_test = nan;
        
        [Mx,My,N_tot,d]=size(U);
        MX=[Mx My];
        M=prod(MX);
        dx=dX(1);
        dy=dX(2);
        a = dx*(1:Mx)';
        b = dy*(1:My);
        grid = {a b'};
        clear U
        
        rho0=1e3;
        
    case 'LES_3D_tot_sub_sample_blurred'
        load([ folder_data '3D_LES_sub_sample_total_blurred.mat'])
        
        U(:,:,:,252:end,:)=[];
        N_test = 250;
        if ndims(U) ~= 5
            error('U does not have the right size. It may already be reshaped');
        end
        [Mx, My, Mz, N_tot,d] = size(U);
        MX=[Mx My Mz];
        M=prod(MX);
        a=dX(1)*(1:MX(1))';
        b=dX(2)*(1:MX(2))';
        c=dX(3)*(1:MX(3))';
        grid = {a b c};
        clear U
        
    case 'DNS_2D_incompact3d_re1000'
        load([ folder_data 'data_incompact3d_re1000.mat'])
        
        [Mx,My,N_tot,d]=size(U);
        N_test=N_tot-1;
        MX=[Mx My];
        M=prod(MX);
        dx=dX(1);
        dy=dX(2);
        a = dx*(1:Mx)';
        b = dy*(1:My);
        grid = {a b'};
        clear U
        
        rho0=1e3;
        
    otherwise
        if nargin < 3
            error('unknown data name');
        end
        
        switch type_whole_data
            case {'inc3D_Re300_40dt_blocks', 'inc3D_Re300_40dt_blocks_truncated',...
                    'inc3D_Re300_40dt_blocks_test_basis'}
                folder_data_in_blocks = 'data_test_in_blocks/';
                file_prefix = 'inc40Dt_';
                
                idx_char_idx_block=length(type_whole_data)+1;
                idx_block = type_data(idx_char_idx_block:end);
                
                load([ folder_data folder_data_in_blocks file_prefix idx_block '.mat']);
                
                [Mx,My,N_tot,d]=size(U);
                MX=[Mx My];
                M=prod(MX);
                dx=dX(1);
                dy=dX(2);
                a = dx*(1:Mx)';
                b = dy*(1:My);
                grid = {a b'};
                clear U
                
            case {'small_test_in_blocks', 'small_test_in_blocks_truncated',...
                    'small_test_in_blocks_test_basis'}
                folder_data_in_blocks = 'data_small_test_in_blocks/';
                file_prefix = 'data_test_in_blocks_';
                
                idx_char_idx_block=length(type_whole_data)+1;
                idx_block = type_data(idx_char_idx_block:end);
                
                load([ folder_data folder_data_in_blocks file_prefix idx_block '.mat']);
                
                [Mx,My,N_tot,d]=size(U);
                MX=[Mx My];
                M=prod(MX);
                dx=dX(1);
                dy=dX(2);
                a = dx*(1:Mx)';
                b = dy*(1:My);
                grid = {a b'};
                clear U
                
            case {'inc3D_Re3900_blocks','inc3D_Re3900_blocks_truncated',...
                    'inc3D_Re3900_blocks_test_basis'}
                folder_data_in_blocks = 'folder_data_DNS3900_blurred/';
%                 folder_data_in_blocks = 'folder_data_DNS3900/';
                file_prefix = 'file_DNS3900_';
                
                idx_char_idx_block=length(type_whole_data)+1;
                idx_block = type_data(idx_char_idx_block:end);
                
                load([ folder_data folder_data_in_blocks file_prefix idx_block '.mat']);
                
                if ndims(U) ~= 5
                    error('U does not have the right size. It may already be reshaped');
                end
                [Mx, My, Mz, N_tot,d] = size(U);
                MX=[Mx My Mz];
                M=prod(MX);
                a=dX(1)*(1:MX(1))';
                b=dX(2)*(1:MX(2))';
                c=dX(3)*(1:MX(3))';
                grid = {a b c};
                clear U
                
%             case 'inc3D_Re3900_blocks_truncated'
%                 folder_data_in_blocks = 'folder_data_DNS3900_blurred/';
% %                 folder_data_in_blocks = 'folder_data_DNS3900/';
%                 file_prefix = 'file_DNS3900_';
%                 
%                 idx_char_idx_block=length(type_whole_data)+1;
%                 idx_block = type_data(idx_char_idx_block:end);
%                 
%                 load([ folder_data folder_data_in_blocks file_prefix idx_block '.mat']);
%                 
%                 if ndims(U) ~= 5
%                     error('U does not have the right size. It may already be reshaped');
%                 end
%                 [Mx, My, Mz, N_tot,d] = size(U);
%                 MX=[Mx My Mz];
%                 M=prod(MX);
%                 a=dX(1)*(1:MX(1))';
%                 b=dX(2)*(1:MX(2))';
%                 c=dX(3)*(1:MX(3))';
%                 grid = {a b c};
%                 clear U
%                 
%             case 'inc3D_Re3900_blocks_test_basis'
%                 folder_data_in_blocks = 'folder_data_DNS3900_blurred/';
% %                 folder_data_in_blocks = 'folder_data_DNS3900/';
%                 file_prefix = 'file_DNS3900_';
%                 
%                 idx_char_idx_block=length(type_whole_data)+1;
%                 idx_block = type_data(idx_char_idx_block:end);
%                 
%                 load([ folder_data folder_data_in_blocks file_prefix idx_block '.mat']);
%                 
%                 if ndims(U) ~= 5
%                     error('U does not have the right size. It may already be reshaped');
%                 end
%                 [Mx, My, Mz, N_tot,d] = size(U);
%                 MX=[Mx My Mz];
%                 M=prod(MX);
%                 a=dX(1)*(1:MX(1))';
%                 b=dX(2)*(1:MX(2))';
%                 c=dX(3)*(1:MX(3))';
%                 grid = {a b c};
%                 clear U

            case {'DNS300_inc3d_3D_2017_04_02_blocks',...
                    'DNS300_inc3d_3D_2017_04_02_blocks_truncated',...
                    'DNS300_inc3d_3D_2017_04_02_blocks_test_basis'}
                
                folder_data_in_blocks = ...
                    'folder_DNS300_inc3d_3D_2017_04_02_blocks/';
                file_prefix = 'file_DNS300_inc3d_3D_2017_04_02_';
                
                idx_char_idx_block=length(type_whole_data)+1;
                idx_block = type_data(idx_char_idx_block:end);
                
                load([ folder_data folder_data_in_blocks ...
                    file_prefix idx_block '.mat']);
                
                if ndims(U) ~= 5
                    error('U does not have the right size. It may already be reshaped');
                end
                [Mx, My, Mz, N_tot,d] = size(U);
                MX=[Mx My Mz];
                M=prod(MX);
                a=dX(1)*(1:MX(1))';
                b=dX(2)*(1:MX(2))';
                c=dX(3)*(1:MX(3))';
                grid = {a b c};
                clear U
                
            case {'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks',...
                    'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_truncated',...
                    'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_test_basis'}
                
                folder_data_in_blocks = ...
                    'folder_DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks/';
                file_prefix = 'file_DNS300_inc3d_3D_2017_04_02_'
                
                idx_char_idx_block=length(type_whole_data)+1;
                idx_block = type_data(idx_char_idx_block:end);
                
                load([ folder_data folder_data_in_blocks ...
                    file_prefix idx_block '.mat']);
                
                if ndims(U) ~= 5
                    error('U does not have the right size. It may already be reshaped');
                end
                [Mx, My, Mz, N_tot,d] = size(U);
                MX=[Mx My Mz];
                M=prod(MX);
                a=dX(1)*(1:MX(1))';
                b=dX(2)*(1:MX(2))';
                c=dX(3)*(1:MX(3))';
                grid = {a b c};
                clear U
                
                
            case {'turb2D_blocks', 'turb2D_blocks_truncated',...
                    'turb2D_blocks_test_basis'}
                folder_data_in_blocks = 'turb2D_blocks/';
                file_prefix = 'turb2D_';
                
                idx_char_idx_block=length(type_whole_data)+1;
                idx_block = type_data(idx_char_idx_block:end);
                
                load([ folder_data folder_data_in_blocks file_prefix idx_block '.mat']);
                dt = 24*3600;
                [Mx,My,N_tot,d]=size(U);
                MX=[Mx My];
                M=prod(MX);
                dX = model.grid.dX;
                dx=dX(1);
                dy=dX(2);
                a = dx*(1:Mx)';
                b = dy*(1:My);
                grid = {a b'};
                nu = model.advection.HV.val;
                forcing = model.advection.forcing;
                clear U
                
            case {'test2D_blocks', 'test2D_blocks_truncated',...
                    'test2D_blocks_test_basis'}
                folder_data_in_blocks = 'test2D_blocks/';
                file_prefix = 'test2D_';
                
                idx_char_idx_block=length(type_whole_data)+1;
                idx_block = type_data(idx_char_idx_block:end);
                
                load([ folder_data folder_data_in_blocks file_prefix idx_block '.mat']);
                dt = 24*3600; % A VERIFIER
                [Mx,My,N_tot,d]=size(U);
                MX=[Mx My];
                M=prod(MX);
                dX = model.grid.dX;
                dx=dX(1);
                dy=dX(2);
                a = dx*(1:Mx)';
                b = dy*(1:My);
                grid = {a b'};
                nu = model.advection.HV.val;
                forcing = model.advection.forcing; %BETA
                clear U
                
            case {'DNS100_inc3d_2D_2018_11_16_blocks_truncated',...
                    'DNS100_inc3d_2D_2018_11_16_blocks', ...
                    'DNS100_inc3d_2D_2018_11_16_blocks_test_basis'}
                
                folder_data_in_blocks = 'folder_DNS100_inc3d_2D_2018_11_16_blocks/';
                file_prefix = 'file_DNS100_inc3d_2018_11_16_run_';
                
                idx_char_idx_block=length(type_whole_data)+1;
                idx_block = type_data(idx_char_idx_block:end);
                
                load([ folder_data folder_data_in_blocks file_prefix idx_block '.mat']);
                
                [Mx,My,N_tot,d]=size(U);
                MX=[Mx My];
                M=prod(MX);
                dx=dX(1);
                dy=dX(2);
                a = dx*(1:Mx)';
                b = dy*(1:My);
                grid = {a b'};
                U=reshape(U,[Mx*My N_tot d]);
                clear U
                
        end
        
        rho0=1e3;
        N_test=nan;
end


%% Grid
% d=size(U,ndims(U));
MX=zeros(1,d); % Numbers of space steps in each directions
for k=1:d
    grid_temp= grid{k};
    MX(k) = length(grid_temp);
end

%% Parameters for the b(t) coefficients calculation
if normalized
    viscosity = 1/Re;
else
    viscosity=nu;
end
if modified_Re
    Re = 1/viscosity;
    Re_true = Re;
    
    % Make the randomness reproductible
    stream = RandStream.getGlobalStream;
    reset(stream);
    
    Re_modif = Re * ( 1 + 0.2 * rand );
    
    if nargin < 3 || eval(idx_block)==1
        fprintf(['Reynolds number has been modified. Real value = ' ...
            num2str(Re_true) ' . New value = ' num2str(Re_modif) '\n']);
    end
    viscosity = 1/Re_modif;
    param.viscosity_true = 1/Re_true;
end
param.dX=dX;
param.MX=MX;
param.M=prod(MX);
param.dt=dt;
if ~exist('rho0','var')
    rho0=1e3;
end
param.rho0=rho0;
param.normalized=normalized;
param.viscosity=viscosity;
param.type_data=type_data;
folder_data=folder_data_ref;
param.folder_data=folder_data;
param.grid=grid;
param.N_test=N_test;
param.N_tot=N_tot;
param.d=d;
% BETA
if exist('model') && (isfield(model,'advection') && isfield(model.advection, 'forcing'))
    param.forcing = forcing;
end
end