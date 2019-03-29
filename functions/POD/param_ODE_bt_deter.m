function [I,L,C, param] = param_ODE_bt_deter(phi_m_U, param, grid)
% Compute the vector I, matrix L and 3-D array C, which define the evolution
% equation of the temporal POD coefficients of the velocity field
% in the case of the classical Navier-Stokes equation
%
% m is the number of modes
% d ( = 2 or 3 ) is the dimension of the velocity field
% - phi ( M x m x d ) the first m spacial modes with their d coefficients in the M points of
% the grid
% - m_U ( M x 1 x d ) the time mean with its d coefficients in the M points of
% the grid
% lambda ( m ) the first m singular values of the POD
% - a ( M x d x d )
% - param ( 2 ) structure wit usefull parmeters like the viscosity and the
% derivation method
% - grid = [ x y (z)] where x, y, z are vectors, without
% repetitions, which define the grid. There are M point
%

%% Pre-treatement

% Spatial modes
if ischar(phi_m_U)
    file_phi=phi_m_U;
    load(phi_m_U,'phi_m_U');
end
% Get size
[~,m,d]=size(phi_m_U);
m=m-1;
% Energy of Chronos
lambda=param.lambda;

% Size tests
if ~( all(size(lambda)== [m 1])...
        && isstruct(param) && all(size(param) == [1 1]) ...
        && iscell(grid) && length(grid) == d )
    error('there is a size problem in the inputs');
end

% Get viscosity parameter
nu = param.viscosity;

% Get data about the grid
dX=param.dX; % Space steps
MX=param.MX; % Numbers of space steps in each directions

% Coding trick
idx='';
for k_dim=1:d
    idx = [idx ',:'];
end

%% Reshape spatial modes
% The treatement of time mean of U (m_U) is done with the treatement of the modes phi
% The form of the arrays should be adapted to the form of the grid to make
% the derivations easier
phi_m_U = permute(phi_m_U,[2 3 1]); % (m+1) x d x M
% (m+1)(number of modes+1) x d(number of velocity components) x M(space)
phi_m_U = reshape(phi_m_U, [m+1 d MX]); % (m+1) x d x Mx x My (x Mz)

%% Compute C
if param.big_data
    % Initialization
    C=zeros(m+1,m+1,m);
    
    for p=1:m+1 % loop on modes
        % Reload phi
        load(file_phi,'phi_m_U');
        phi_m_U=phi_m_U(:,p,:);
        phi_m_U = permute(phi_m_U,[2 3 1]); % (1) x d x M
        % (m+1)(number of modes+1) x d(number of velocity components) x M(space)
        phi_m_U = reshape(phi_m_U, [1 d MX]); % (1) x d x Mx x My (x Mz)
        
        % Gradient of the modes phi % (1) x d x Mx x My (x Mz) x d
        dphi_m_U = gradient_mat(phi_m_U,dX);
        clear phi_m_U
        dphi_m_U = permute(dphi_m_U, [1 ndims(dphi_m_U)+1 ndims(dphi_m_U) 3:2+d 2]);
        % (1) x 1 x d! x Mx x My (x Mz) x d
        
        for q=1:m+1 % loop on modes
            % Reload phi
            load(file_phi,'phi_m_U');
            phi_m_U=phi_m_U(:,q,:);
            phi_m_U = permute(phi_m_U,[4 2 3 1]); % 1 x (1) x d x M
            % (m+1)(number of modes+1) x d(number of velocity components) x M(space)
            phi_m_U = reshape(phi_m_U, [1 1 d MX]); % 1 x (1) x d x Mx x My (x Mz)
            
            % Compute phi_m_U_{j} nabla' phi_m_U_{i}
            c_integrand = bsxfun(@times, phi_m_U, dphi_m_U); % (1) x (1) x d! x Mx x My (x Mz) x d
            clear phi_m_U
            
            c_integrand = permute(sum(c_integrand,3),[1 2 4:ndims(c_integrand) 3]); % (1) x (1) x Mx x My (x Mz) x d
            
            % Projection on free divergence space to remove the unknown
            % pressure term
            if param.eq_proj_div_free
                c_integrand = reshape(c_integrand,[prod(MX) 1 d]);
                c_integrand = c_integrand - proj_div_propre(c_integrand,MX);
                c_integrand = reshape(c_integrand,[1 1 MX d]);
            end
            
            for k = 1:m
                % Reload phi
                load(file_phi,'phi_m_U');
                phi=phi_m_U(:,k,:); clear phi_m_U
                phi = permute(phi,[4 2 1 3]); % 1 x (m) x M x d
                % (m+1)(number of modes+1) x d(number of velocity components) x M(space)
                phi = reshape(phi, [1 1 MX d]); % 1 x (1) x Mx x My (x Mz) x d
                
                % Projection on phi_{k}
                c_temp = c_integrand.* phi;% (1) x (1) x Mx x My (x Mz) x d
                clear phi
                c_temp = sum(c_temp, ndims(c_temp)); % (1) x (1) x Mx x My (x Mz)
                % Integration on the space
                C(p,q,k) = integration_mat(c_temp, dX,param.big_data); % (1) x (1) x 1
                clear c_temp;
            end
            clear c_integrand;
            
        end
    end
    
else %% Small data used
    % Compute the gradient, Laplacian and other differial results
    
    % Gradient of the modes phi % (m+1) x d x Mx x My (x Mz) x d
    dphi_m_U = gradient_mat(phi_m_U,dX);
    
    % Compute phi_m_U_{j} nabla' phi_m_U_{i}
    
    % phi_m_U (m+1) x d! x Mx x My (x Mz)
    % dphi_m_U (m+1) x d x Mx x My (x Mz) x d!
    phi_m_U = permute(phi_m_U, [ndims(phi_m_U)+1 1:ndims(phi_m_U)]);
    % 1 x (m+1) x d! x Mx x My (x Mz)
    dphi_m_U = permute(dphi_m_U, [1 ndims(dphi_m_U)+1 ndims(dphi_m_U) 3:2+d 2]);
    % (m+1) x 1 x d! x Mx x My (x Mz) x d
    c_integrand = bsxfun(@times, phi_m_U, dphi_m_U); % (m+1) x (m+1) x d! x Mx x My (x Mz) x d
    clear dphi_m_U;
    c_integrand = squeeze(sum(c_integrand,3)); % (m+1) x (m+1) x Mx x My (x Mz) x d
    
    % Projection on free divergence space to remove the unknown
    % pressure term
    if param.eq_proj_div_free
        c_integrand = reshape(c_integrand,[ (m+1)^2 prod(MX) d]);% (m+1)^2 x M x d
        c_integrand = multitrans(c_integrand);% M x (m+1)^2 x d
        c_integrand = c_integrand - proj_div_propre(c_integrand,MX);
        c_integrand = multitrans(c_integrand);% (m+1)^2 xM x  d
        c_integrand = reshape(c_integrand,[(m+1) (m+1) MX d]);
    end
    
    % Remove time average m_U
    eval(['phi=phi_m_U(1,1:end-1,:' idx ');']);
    clear phi_m_U;
    phi = permute(phi, [1 2 4:ndims(phi) 3]); % 1 x m+1 x Mx x My (x Mz) x d
    
    C = zeros(m+1,m+1,m);
    for k = 1:m % loop on modes
        % Projection on phi_{k}
        eval(['c_temp = bsxfun(@times, c_integrand, phi(1,k' idx ',:));']);
        % (m+1) x (m+1) x Mx x My (x Mz) x d
        c_temp = squeeze(sum(c_temp, ndims(c_temp))); % (m+1) x (m+1) x Mx x My (x Mz)
        % Integration on the space
        C(:,:,k) = integration_mat(c_temp, dX,param.big_data); % (m+1) x (m+1) x 1
        clear c_temp;
    end
    clear c_integrand phi;
    
end

%% Compute L

if param.big_data
    % Initialization
    L=zeros(m,m);
    
    for p=1:m % loop on modes
        % Reload phi
        load(file_phi,'phi_m_U');
        phi=phi_m_U(:,p,:);clear phi_m_U
        phi = permute(phi,[2 3 1]); % (1) x d x M
        % (m+1)(number of modes+1) x d(number of velocity components) x M(space)
        phi = reshape(phi, [1 d MX]); % (1) x d x Mx x My (x Mz)
        
        % Molecular viscosity
        if nu~=0
            % Laplacian of the modes phi % (m+1) x d x Mx x My (x Mz)
            d2phi = laplacian_mat(phi,dX);
            % size(d2phi) = (1) x d x Mx x My (x Mz)
            t4 = - nu * permute(d2phi ,[1 ndims(d2phi)+1 2:ndims(d2phi)]);
            % (1) x 1 x d x Mx x My (x Mz)
            clear d2phi;
        else
            t4=0;
        end
        
        % Adding liner forces
        L_p = t4;
        clear t4;
        
        for q=1:m % loop on modes
            % Reload phi
            load(file_phi,'phi_m_U');
            phi=phi_m_U(:,q,:); clear phi_m_U
            phi = permute(phi,[4 2 3 1]); % 1 x (m) x d x M
            % (m+1)(number of modes+1) x d(number of velocity components) x M(space)
            phi = reshape(phi, [1 1 d MX]); % 1 x (1) x d x Mx x My (x Mz)
            
            % L (m) x 1 x d x Mx x My (x Mz)
            Lpq = bsxfun(@times,L_p,phi); % (m) x m x d x Mx x My (x Mz) (x N)
            clear phi;
            Lpq= permute(sum(Lpq,3),[1 2 4:ndims(Lpq) 3]); % (m+1) x m x Mx x My (x Mz) x 1 x (x N)
            Lpq = integration_mat(Lpq, dX,param.big_data); % (m+1)or(m+1)*N  x m
            L(p,q)=Lpq;
        end
    end
    
else %% Small data used
    
    % Reload phi
    load(file_phi,'phi_m_U'); % M x m+1 x d
    % The form of the arrays should be adapted to the form of the grid to make
    % the derivations easier
    phi = permute(phi_m_U(:,1:end-1,:),[2 3 1]); % (m) x d x M
    clear phi_m_U
    % (m+1)(number of modes+1) x d(number of velocity components) x M(space)
    phi = reshape(phi, [m d MX]); % (m) x d x Mx x My (x Mz)
    
    % Molecular viscosity
    if nu~=0
        % Laplacian of the modes phi % (m+1) x d x Mx x My (x Mz)
        d2phi = laplacian_mat(phi,dX);
        % size(d2phi) = (m) x d x Mx x My (x Mz)
        t4 = - nu * permute(d2phi ,[1 ndims(d2phi)+1 2:ndims(d2phi)]);
        % (m) x 1 x d x Mx x My (x Mz)
        clear d2phi;
    else
        t4=0;
    end
    
    % Adding liner forces
    L = t4;
    clear t4;
    
    % Projection on phi_{j}
    % phi  m x d x Mx x My (x Mz)
    phi=permute(phi,[ndims(phi)+1 1:ndims(phi)]); % 1 x m x d x Mx x My (x Mz)
    % L (m) x 1 x d x Mx x My (x Mz)
    L = bsxfun(@times,L,phi); % (m) x m x d x Mx x My (x Mz) (x N)
    clear phi;
    L= permute(sum(L,3),[1 2 4:ndims(L) 3]); % (m+1) x m x Mx x My (x Mz) x 1 x (x N)
    L = integration_mat(L, dX,param.big_data); % (m+1)or(m+1)*N  x m
    
end

%% Final calculation (due to remove constant in time part of Navier Stokes equation)
I=zeros(m,1);
for i=1:m
    I=I-lambda(i)*squeeze(C(i,i,:));
end
L = L + squeeze(C(m+1,1:m,:))+squeeze(C(1:m,m+1,:)) ; % m x m
% Keep the unused part of C for other off-line calculation
param.C_deter_residu=squeeze(C(m+1,1:m,:))+squeeze(C(1:m,m+1,:));

C = C(1:end-1,1:end-1,:); % m x m x m

end