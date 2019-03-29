function S = proj_div_propre(S,MX,dX, periodic_condition)
% proj_div_propre = (Id - proj_free_div)
% with proj_free_div forcing the constraint div(S) = 0
% Size(S) = M x N x d
% M is the number of point in the space
% N is arbitrary (for instance, number of time step)
% MX = [Mx My] or [ Mx My Mz] is the number of point in respectively the
% axis Ox, Oy and Oz
% d = 2 or 3 is the dimension
%
if nargin < 4
    periodic_condition = false;
end

bool_dealiasing = true;
[M,N,d]=size(S);
bool_real = all(all(all(isreal(S))));

%% Extension
S=reshape(S,[M N*d])';
% Extrapole the function to zero outside the domain in order to be able to
% use Dirichlet conditions
if ~periodic_condition
    [S,MX]=extension_to_zero(S,MX);
end


M=prod(MX);
S=reshape(S',[M N d]);

%% Fourrier transform
S = reshape(S,[MX N d]); % Mx x My (x Mz) x N x d

for i=1:d
    S = fft(S,[],i);
end
MX=size(S);
MX=MX(1:d);

%% Projection on the subspace of free divergence matrix

%% (unstable) Spacial frequency k
k=zeros([MX d]);
PX=ceil(MX/2);
if ~mod(MX(1),2)
    nx=[ 0:(PX(1)-1) 0 (1-PX(1)):-1] ;
else
    nx=[ 0:(PX(1)-1) (1-PX(1)):-1] ;
end
kx=(2*pi)/(MX(1)*dX(1))*nx ;
if ~mod(MX(2),2)
    ny=[ 0:(PX(2)-1) 0 (1-PX(2)):-1] ;
else
    ny=[ 0:(PX(2)-1) (1-PX(2)):-1] ;
end
ky=(2*pi)/(MX(2)*dX(2))*ny;
if d==2
    [kx,ky]=ndgrid(kx,ky);
    k(:,:,1)=kx;
    k(:,:,2)=ky;
else
    if ~mod(MX(3),2)
        nz=[ 0:(PX(3)-1) 0 (1-PX(3)):-1] ;
    else
        nz=[ 0:(PX(3)-1) (1-PX(3)):-1] ;
    end
    kz =(2*pi)/(MX(3)*dX(3))*nz;
    [kx,ky,kz]=ndgrid(kx,ky,kz);
    k(:,:,:,1)=kx;
    k(:,:,:,2)=ky;
    k(:,:,:,3)=kz;
end

norm_k_2 = sum( k.^2, d+1); % Mx x My x Mz x 1

%% Anti aliasing mask
alpha = 36.;
order = 19.;
maskx = exp(-alpha*( (2./MX(1)).*abs(nx) ).^order);
masky = exp(-alpha*( (2./MX(2)).*abs(ny) ).^order);
if d == 2
    mask = maskx'*masky;
    if ~mod(MX(1),2)
        mask(PX(1)+1,:) = 0.; %de-alias the single high freq
    end
    if ~mod(MX(2),2)
        mask(:,PX(2)+1) = 0.;
    end
else
    maskz = exp(-alpha*( (2./MX(3)).*abs(nz) ).^order);
    mask = bsxfun( @times, maskx' * masky , ...
        permute( maskz, [1 3 2] ) ) ;
    if ~mod(MX(1),2)
        mask(PX(1)+1,:,:) = 0.; %de-alias the single high freq
    end
    if ~mod(MX(2),2)
        mask(:,PX(2)+1,:) = 0.;
    end
    if ~mod(MX(3),2)
        mask(:,:,PX(3)+1) = 0.;
    end
end

%% Projection operator A
% A_{i,j} = k_i k_j / ||k||_2^2
% Mx x My x Mz x d x d
A = bsxfun(@times, k, permute(k,[(1:d) d+2 d+1]) ); % Mx x My x Mz x d x d
% norm_k_2 = sum( k.^2, d+1); % Mx x My x Mz x 1
A = bsxfun(@times, 1./norm_k_2, A); % Mx x My x Mz x d x d
if bool_dealiasing
    A = bsxfun(@times, mask , A); % Mx x My x Mz x d x d
end

% The operator do not modify the constant value
if d==2
    A(1,1,:,:)=0;
    if ~mod(MX(1),2)
        A(PX(1)+1,:,:,:)=0;
    end
    if ~mod(MX(2),2)
        A(:,PX(2)+1,:,:)=0;
    end
else
    A(1,1,1,:,:)=0;
    if ~mod(MX(1),2)
        A(PX(1)+1,:,:,:,:)=0;
    end
    if ~mod(MX(2),2)
        A(:,PX(2)+1,:,:,:)=0;
    end
    if ~mod(MX(3),2)
        A(:,:,PX(3)+1,:,:)=0;
    end
end

%% Application of the operator
A = permute(A, [1:d d+3 d+1 d+2]); % Mx x My x Mz x 1 x d x d
S = bsxfun( @times, S, A); % Mx x My x Mz x N x d x d
S = squeeze(sum(S, d+2)); % Mx x My x Mz x N x d

%% Inverse Fourrier transform
for i=1:d
    S = ifft(S,[],i);
end
MX=size(S);
MX=MX(1:d);

if bool_real
    S = real(S);
end
S = reshape(S,[M N d]); % M x N x d

%% Crop to remove the extrapolation
S=reshape(S,[M N*d])'; % N*d x M
if ~periodic_condition
    [S,MX]=crop_extension(S,MX);
end


M=prod(MX);
S=reshape(S',[M N d]);

