function S = proj_div_propre(S,MX)
% proj_div_propre = (Id - proj_free_div)
% with proj_free_div forcing the constraint div(S) = 0
% Size(S) = M x N x d
% M is the number of point in the space
% N is arbitrary (for instance, number of time step)
% MX = [Mx My] or [ Mx My Mz] is the number of point in respectively the
% axis Ox, Oy and Oz
% d = 2 or 3 is the dimension
%

[M,N,d]=size(S);
bool_real = all(all(all(isreal(S))));

%% Extension
S=reshape(S,[M N*d])';
% Extrapole the function to zero outside the domain in order to be able to
% use Dirichlet conditions

% [S,MX]=extension_to_zero(S,MX);
S=S';

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

% Spacial frequency k
k=zeros([MX d]);
PX=ceil(MX/2);
if ~mod(MX(1),2)
    kx=1/(MX(1))*[ 0:(PX(1)-1) 0 (1-PX(1)):-1] ;
else
    kx=1/(MX(1))*[ 0:(PX(1)-1) (1-PX(1)):-1] ;
end
if ~mod(MX(2),2)
    ky=1/(MX(2))*[ 0:(PX(2)-1) 0 (1-PX(2)):-1] ;
else
    ky=1/(MX(2))*[ 0:(PX(2)-1) (1-PX(2)):-1] ;
end
if d==2
    [kx,ky]=ndgrid(kx,ky);
    k(:,:,1)=kx;
    k(:,:,2)=ky;
else
    if ~mod(MX(3),2)
        kz=1/(MX(3))*[ 0:(PX(3)-1) 0 (1-PX(3)):-1] ;
    else
        kz=1/(MX(3))*[ 0:(PX(3)-1) (1-PX(3)):-1] ;
    end
    [kx,ky,kz]=ndgrid(kx,ky,kz);
    k(:,:,:,1)=kx;
    k(:,:,:,2)=ky;
    k(:,:,:,3)=kz;
end

norm_k_2 = sum( k.^2, d+1); % Mx x My x Mz x 1

% Spacial frequency k
k=zeros([MX d]);
PX=ceil(MX/2);
% PX=MX/2;
% meth_anti_alias='none';
meth_anti_alias='deriv_LowPass';
if ~mod(MX(1),2)
    kx=1/(MX(1))*[ 0:(PX(1)-1) 0 (1-PX(1)):-1] ;
else
    kx=1/(MX(1))*[ 0:(PX(1)-1) (1-PX(1)):-1] ;
end
if ~mod(MX(2),2)
    ky=1/(MX(2))*[ 0:(PX(2)-1) 0 (1-PX(2)):-1] ;
else
    ky=1/(MX(2))*[ 0:(PX(2)-1) (1-PX(2)):-1] ;
end
if strcmp(meth_anti_alias,'deriv_LowPass')
    kx = kx .* fct_unity_approx5(MX(1));
    ky = ky .* fct_unity_approx5(MX(2));
end
if d==2
    [kx,ky]=ndgrid(kx,ky);
    k(:,:,1)=kx;
    k(:,:,2)=ky;
else
    if ~mod(MX(3),2)
        kz=1/(MX(3))*[ 0:(PX(3)-1) 0 (1-PX(3)):-1] ;
    else
        kz=1/(MX(3))*[ 0:(PX(3)-1) (1-PX(3)):-1] ;
    end
    if strcmp(meth_anti_alias,'deriv_LowPass')
        kz= kz .* fct_unity_approx5(MX(3));
    end
    [kx,ky,kz]=ndgrid(kx,ky,kz);
    k(:,:,:,1)=kx;
    k(:,:,:,2)=ky;
    k(:,:,:,3)=kz;
end

% Projection operator A
% A_{i,j} = \delta_{i,j} - k_i k_j / ||k||_2^2
% Mx x My x Mz x d x d
A = bsxfun(@times, k, permute(k,[(1:d) d+2 d+1]) ); % Mx x My x Mz x d x d
% norm_k_2 = sum( k.^2, d+1); % Mx x My x Mz x 1
A = bsxfun(@times, 1./norm_k_2, A); % Mx x My x Mz x d x d

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

% meth_anti_alias='fct_LowPass';
% 
% if strcmp(meth_anti_alias,'fct_LowPass')
%     A= bsxfun(@times, fct_unity_approx5(MX(1))',A);
%     A= bsxfun(@times, fct_unity_approx5(MX(2)) ,A);
%     if d==3
%         A= bsxfun(@times, permute(fct_unity_approx5(MX(3)),[1 3 2]) ,A);
%     end
% end

% Application of the operator
A = permute(A, [1:d d+3 d+1 d+2]); % Mx x My x Mz x 1 x d x d
S = bsxfun( @times, S, A); % Mx x My x Mz x N x d x d
S = squeeze(sum(S, d+2)); % Mx x My x Mz x N x d

% meth_anti_alias='fct_LowPass';
% 
% if strcmp(meth_anti_alias,'fct_LowPass')
%     S= bsxfun(@times, fct_unity_approx5(MX(1))',S);
%     S= bsxfun(@times, fct_unity_approx5(MX(2)) ,S);
%     if d==3
%         S= bsxfun(@times, permute(fct_unity_approx5(MX(3)),[1 3 2]) ,S);
%     end
% end

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

% [S,MX]=crop_extension(S,MX);
S=S';

M=prod(MX);
S=reshape(S',[M N d]);

