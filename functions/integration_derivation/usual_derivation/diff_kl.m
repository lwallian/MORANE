function d2a = diff_kl(a,dX)
% a must have the size :  m x d x d x d x Mx x My (x Mz)
% Compute d2a(:,:,k,l,x_1,x_2(,x_3))= d_{x_k} d_{x_l} a(:,:,k,l,x_1,x_2(,x_3))
% ( derivation along the axis x_k and x_l
% Then, the results is sumed along the 3rd and 4th dimension
% The result have the size m x d x Mx x My (x Mz)

siz = size(a); % = [m x d x d x d x Mx x My (x Mz)]
d=length(siz)-4;

idx='';
for k_dim=1:d
    idx = [idx ',:'];
end

d2a=zeros(siz);

for l=1:d
    eval(['d2a(:,:,:,l' idx ')= diff_l(a(:,:,:,l' idx '),l,dX);']);
end
d2a = permute(d2a,[1 2 4 3 5:ndims(d2a)]);
for k=1:d
    eval(['d2a(:,:,:,k' idx ')= diff_l(d2a(:,:,:,k' idx '),k,dX);']);
end
d2a = permute(d2a,[1 2 4 3 5:ndims(d2a)]);

d2a = sum(sum(d2a,4),3);
d2a = permute(d2a, [1 2 5:ndims(d2a) 3 4]);
