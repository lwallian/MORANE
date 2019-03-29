function [I,L,C] = param_ODE_bt_sto(a, param, grid)
% Compute the vector I, matrix L and 3-D array C, which define the evolution
% equation of the temporal POD coefficients of the velocity field
% in the case of stochastic Navier-Stokes model
%
% m is the number of modes
% d ( = 2 or 3 ) is the dimension of the velocity field
% - phi ( M x m x d ) the first m spacial modes with their d coefficients in the M points of
% the grid
% - m_U ( M x 1 x d ) the time mean with its d coefficients in the M points of
% the grid
% lambda ( m ) the first m singular values of the POD
% - a : name of folder wheere there are spatial modes of a
% - param ( 2 ) structure wit usefull parmeters like the viscosity and the
% derivation method
% - grid = [ x y (z)] where x, y, z are vectors, without
% repetitions, which define the grid. There are M point
%

%% Pre treatement

% Model for the variance tensor a
if param.a_time_dependant && strcmp(param.type_filter_a,'b_i')
    nb_modes_z = param.nb_modes+1;
elseif ~ param.a_time_dependant
    nb_modes_z =1;
else
    error('not coded or depreciated');
end

% Spatial modes
if ischar(a)
    load(a,'phi_m_U');
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

% Get data about the grid
dX=param.dX; % Space steps
MX=param.MX; % Numbers of space steps in each directions

% Coding trick
idx='';
for k_dim=1:d
    idx = [idx ',:'];
end

%% Compute C
if param.big_data
    % Initialization
    C=zeros(m+1,nb_modes_z,m);
    
    for p=1:m+1 % loop on modes
        % Reload phi
        load(a,'phi_m_U');
        phimU=phi_m_U(:,p,:);clear phi_m_U
        
        for q=1:nb_modes_z % loop on modes
            
            % First term of the tensor
            % Reload z_q(i,j)
            %             load(a,'z');
            load(param.name_file_diffusion_mode);
            
            
            z=z(:,q,:,:);
            z_phi_m_U = permute(z, [1 5 6 3 4 2]); %  M x 1 x 1 x d x d x (1)
            clear z;
            z_phi_m_U = bsxfun(@times,z_phi_m_U, phimU); %  M x (1)(phi) x (d)(phi) x d x d x (1)(z)
            
            % The form of the arrays should be adapted to the form of the grid to make
            % the derivations easier
            z_phi_m_U = permute(z_phi_m_U,[2 3 4 5 1 6]); % (1)(phi) x (d)(phi) x d x d x M x (1)(z)
            z_phi_m_U = reshape(z_phi_m_U, [1 d d d MX 1]);
            % (1)(phi) x (d)(phi) x d x d x Mx x My (x Mz) x (1)(z)
            
            % Compute differials
            z_phi_m_U = permute(z_phi_m_U,[1 (d+5) 2:(d+4)]);
            % (1)(phi) x (1)(z) x (d)(phi) x d x d x Mx x My (x Mz)
            z_phi_m_U = reshape(z_phi_m_U,[(1)^2 d d d MX]);% (1)(phi)*(1)(z) x (d)(phi) x d x d x Mx x My (x Mz)
            d2z_phi_m_U = diff_kl(z_phi_m_U,dX);
            % (1)(phi)*(1)(z) x  (d)(phi) x Mx x My (x Mz)
            clear z_phi_m_U
            d2z_phi_m_U = permute(d2z_phi_m_U, [1 d+3 3:d+2 2]);% (1)(phi)*(1)(z)x 1 x Mx x My (x Mz)x  (d)(phi)
            
            % Secon term of the tensor
            % Reload z_q(i,j)
            %             load(a,'z');
            load(param.name_file_diffusion_mode);
            z=z(:,q,:,:);
            z = permute(z,[2 5 6 3 4 1 ]); % (1) x 1 x d x d x M
            z = reshape(z, [(1) 1 d d MX]); % (1)(z) x 1 x d x d x Mx x My (x Mz)
            
            % Compute differials
            d2z=diff_kl(z,dX);% (1)(z) x 1 x Mx x My (x Mz)
            clear z;
            
            d2z = bsxfun(@times, d2z, reshape(phimU ,[1 1 MX d]) );% (1)(z) x 1 x Mx x My (x Mz) x d
            
            % Add the two terms of the tensor
            c_integrand = -1/2*(d2z_phi_m_U - d2z); % 1 x 1 x Mx x My (x Mz) x d
            clear d2z_phi_m_U d2z;
            
            % Projection on free divergence space to remove the unknown
            % pressure term
            c_integrand = reshape(c_integrand,[prod(MX) 1 d]);
            if strcmp(param.type_data, 'turb2D_blocks_truncated')
                c_integrand = c_integrand - proj_div_propre(c_integrand,MX,dX, true);
            else
                c_integrand = c_integrand - proj_div_propre(c_integrand,MX,dX, false);
            end
            c_integrand = reshape(c_integrand,[1 1 MX d]);
            
            for k = 1:m
                % Reload phi
                load(a,'phi_m_U');
                phi=phi_m_U(:,k,:); clear phi_m_U
                phi = permute(phi,[4 2 1 3]); % 1 x (1) x M x d
                % (m+1)(number of modes+1) x d(number of velocity components) x M(space)
                phi = reshape(phi, [1 1 MX d]); % 1 x (1) x Mx x My (x Mz) x (d)
                
                % Projection on phi_{k}
                c_temp = c_integrand.* phi;% (1) x (1) x Mx x My (x Mz) x (d)
                clear phi
                c_temp = sum(c_temp, 3+d); % (1) x (1) x Mx x My (x Mz)
                % Integration on the space
                C(p,q,k) = C(p,q,k) + integration_mat(c_temp, dX,param.big_data); % (1) x (1) x 1
                clear c_temp;
            end
            clear c_integrand;
            
        end
    end
    
else %% Small data used
    % Compute a_{k,l} * phi
    if ischar(a)
        a_file=a;
        %         load(a,'z');
        load(param.name_file_diffusion_mode);
    end
    z_phi_m_U = permute(z,[1 5 6 3 4 2]); %  M x 1 x 1 x d x d x (nb_modes_z)
    clear z;
    z_phi_m_U = bsxfun(@times,z_phi_m_U, phi_m_U); %  M x (m+1)(phi) x d(phi) x d x d x (nb_modes_z)(z)
    
    
    % The form of the arrays should be adapted to the form of the grid to make
    % the derivations easier
    phi_m_U = permute(phi_m_U,[2 3 1]); % (m+1) x d x M
    % (m+1)(number of modes+1) x d(number of velocity components) x M(space)
    phi_m_U = reshape(phi_m_U, [m+1 d MX]); % (m+1) x d x Mx x My (x Mz)
    
    z_phi_m_U = permute(z_phi_m_U,[2 3 4 5 1 6]); % (m+1)(phi) x d(phi) x d x d x M x (nb_modes_z)(z)
    z_phi_m_U = reshape(z_phi_m_U, [m+1 d d d MX nb_modes_z]);
    % (m+1)(phi) x d(phi) x d x d x Mx x My (x Mz) x (nb_modes_z)(z)
    
    if nb_modes_z >1
        z_phi_m_U = permute(z_phi_m_U,[1 ndims(z_phi_m_U) 2:(ndims(z_phi_m_U)-1)]);
    else
        z_phi_m_U = permute(z_phi_m_U,[1 ndims(z_phi_m_U)+1 2:(ndims(z_phi_m_U))]);
    end
    z_phi_m_U = reshape(z_phi_m_U,[(m+1)*nb_modes_z d d d MX]);% (m+1)(phi)*(nb_modes_z)(z) x d(phi) x d x d x Mx x My (x Mz)
    
    
    %% Compute differials
    
    % First term of the tensor
    % Compute derivatives
    if numel(z_phi_m_U)>1e9
        for k=size(z_phi_m_U,1):-1:1
            eval(['d2z_phi_m_U(k,:' idx ') = diff_kl(z_phi_m_U(k,:,:,:' ...
                idx '),dX);']);
            % (m+1)(phi)*(nb_modes_z)(z) x d x Mx x My (x Mz)
            eval(['z_phi_m_U(k,:,:,:' idx ')=[];']);
        end
    else
        d2z_phi_m_U = diff_kl(z_phi_m_U,dX);
        % (m+1)(phi)*(nb_modes_z)(z) x d(phi) x Mx x My (x Mz)
        clear z_phi_m_U
    end
    
    % Secon term of the tensor
    % Reload z_q(i,j)
    %     load(a_file,'z');
    load(param.name_file_diffusion_mode);
    
    z = permute(z,[5 6 3 4 1 2 ]); % 1 x 1 x d x d x M x (nb_modes_z)
    z = reshape(z, [1 1 d d MX (nb_modes_z)]); % 1 x 1 x d x d x Mx x My (x Mz) x (nb_modes_z)
    if nb_modes_z > 1
        z = permute(z,[ndims(z) 2:(ndims(z)-1) 1]);% (m+1)(z) x 1 x d x d x Mx x My (x Mz)
    else
        z = permute(z,[ndims(z)+1 2:(ndims(z)) 1]);% (m+1)(z) x 1 x d x d x Mx x My (x Mz)
    end
    
    % Compute derivatives
    d2z=diff_kl(z,dX);% (nb_modes_z)(z) x 1 x Mx x My (x Mz)
    clear z;
    
    d2z_phi_m_U = reshape(d2z_phi_m_U,[(m+1) (nb_modes_z) d MX]);% (m+1)(phi) x (nb_modes_z)(z) x d x Mx x My
    d2z_phi_m_U = permute(d2z_phi_m_U,[1 3:(3+d) 2]);% (m+1)(phi) x d(phi) x Mx x My x (nb_modes_z)(z)
    if nb_modes_z > 1
        d2z = permute(d2z,[ndims(d2z)+1 2:(ndims(d2z)) 1]);% 1 x 1 x Mx x My x (nb_modes_z)(z)
    else
        d2z = permute(d2z,[ndims(d2z)+2 2:(ndims(d2z)+1) 1]);% 1 x 1 x Mx x My x (nb_modes_z)(z)
    end
    
    % Projection on Topos
    d2z=bsxfun(@times,d2z, phi_m_U);% (m+1)(phi) x d(phi) x Mx x My (x Mz) x (nb_modes_z)(z)
    % Add the two terms of the tensor
    d2z_phi_m_U = d2z_phi_m_U - d2z; % (m+1)(phi) x d(phi) x Mx x My (x Mz) x (nb_modes_z)(z)
    clear d2z;
    
    
    %% Compute C
    
    % Compute phi_m_U_{j} nabla' phi_m_U_{i}
    % d2z_phi_m_U % (m+1)(phi) x d x Mx x My (x Mz) x (m+1)(z)
    d2z_phi_m_U = permute(d2z_phi_m_U,[1 3+d 3:(2+d) 2]);% (m+1)(phi) x (nb_modes_z)(z) x Mx x My (x Mz) x d
    c_integrand = -1/2 * d2z_phi_m_U;
    clear d2z_phi_m_U;
    
    % Projection on free divergence space to remove the unknown
    % pressure term
    c_integrand = reshape(c_integrand,[(m+1)*nb_modes_z prod(MX) d]);% (m+1)*nb_modes_z x M x d
    c_integrand = multitrans(c_integrand);% M x (m+1)*nb_modes_z x d
    if strcmp(param.type_data, 'turb2D_blocks_truncated')
        c_integrand = c_integrand - proj_div_propre(c_integrand,MX,dX, true);
    else
        c_integrand = c_integrand - proj_div_propre(c_integrand,MX,dX, false);
    end
    c_integrand = multitrans(c_integrand);% (m+1)*nb_modes_z x M x d
    c_integrand = reshape(c_integrand,[m+1 nb_modes_z MX d]);
    
    % % phi m x d! x Mx x My (x Mz)
    phi_m_U = permute(phi_m_U, [ ndims(phi_m_U)+1 1 3:2+d 2]); % 1 x m(k) x Mx x My (x Mz) x d
    
    % Initialization
    C = zeros(m+1,nb_modes_z,m);% c_{i,j,k}
    for k = 1:m % loop on modes
        % Projection on phi_{k}
        eval(['c_temp = bsxfun(@times, c_integrand, phi_m_U(1,k' idx ',:));']);
        % (m+1)(i) x (nb_modes_z)(j) x Mx x My (x Mz) x d
        c_temp = sum(c_temp, ndims(c_temp)); % (m+1)(i) x (nb_modes_z)(j) x Mx x My (x Mz)
        % Integration on the space
        C(:,:,k) = integration_mat(c_temp, dX,param.big_data); % (m+1)(i) x (nb_modes_z)(j) x 1
        clear c_temp;
    end
    clear c_integrand phi_m_U;
end

%% Final calculation (due to remove constant in time part of Navier Stokes equation)
I=zeros(m,1);
if nb_modes_z > 1
    for i=1:m
        I=I-lambda(i)*squeeze(C(i,i,:));
    end
    L = squeeze(C(m+1,1:(nb_modes_z-1),:))+squeeze(C(1:m,nb_modes_z,:)) ; % m x m
    C = C(1:end-1,1:end-1,:); % m x m x m
else
    L = squeeze(C(1:m,nb_modes_z,:)) ; % m x m
    C = zeros(m,m,m);
end

end