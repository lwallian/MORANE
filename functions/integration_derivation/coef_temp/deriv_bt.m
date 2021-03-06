function d_b = deriv_bt(I,L,C, bt)
% Compute the time derivation of bt
% The sizes of the inputs should be :
% - I : m
% - L : m x m
% - C : m x m x m
% - bt : N x m 
% - ord3 (if exist) : m x m x m x m
% - ord4 (if exist) : m x m x m x mx m
% The result has the size : N x m
%

N = size(bt,1);

bt = permute(bt,[2 3 4 1]); % m x 1 x 1 x N

C = bsxfun(@times,bt,C); % m x m x m x N
C = squeeze(sum(C,1)); % m x m x N

bt = permute(bt,[1 2 4 3]); % m x 1 x N
C = bsxfun(@times,bt,C); % m x m x N
C = sum(C,1); % 1 x m x N
C = permute(C,[2 3 1]); % m x N

L = bsxfun(@times,bt,L); % m x m x N
L = sum(L,1); % 1 x m x N
L = permute(L,[2 3 1]); % m x N

d_b = - ( repmat(I,[1,N]) + L + C )' ; % N x m

