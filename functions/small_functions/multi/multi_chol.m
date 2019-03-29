function L = multi_chol(A,type)
% Generalisation of chol(A) if A is an ND array
% Cholesky-Crout algorithm (column by column)
% L' * L = A
%

if nargin < 2
    warning('the symetriy and positivity of matrix is not tested in multi_chol');
    
    idx =[];
    for k=3:ndims(A)
        idx = [idx ',:'];
    end
    
    sA = size(A);
    if sA(1) ~= sA(2)
        error('wrong size');
    end
    L = zeros(sA);
    
    d=sA(1);
    eval(['L(1,1' idx ') = real(sqrt( A(1,1' idx ')));']);
    eval(['test=abs(squeeze(L(1,1' idx ')))<eps;']);
    if any(test(:))
        error('At least one matrix is not invertible');
    end
    eval(['L(2:end,1' idx ') =  A(2:end,1' idx ');']);
    eval(['den = 1./L(1,1' idx ');']);
    eval(['L(2:end,1' idx ') =  bsxfun(@times, L(2:end,1' idx '), den);']);
    for j=2:(d-1)
        eval(['L(j,j' idx ') = real(sqrt( A(j,j' idx ') - multiprod( ' ...
            'L(j,1:(j-1)' idx ') , multitrans ( L(j,1:(j-1)' idx ') ) ) ));']);
        eval(['test=abs(squeeze(L(j,j' idx ')))<eps;']);
        if any(test(:))
            error('At least one matrix is not invertible');
        end
        eval(['L((j+1):end,j' idx ') = A((j+1):end,j' idx ') - multiprod( ' ...
            'L((j+1):end,1:(j-1)' idx ') , multitrans ( L(j,1:(j-1)' idx ') ) );']);
        eval(['den = 1./L(j,j' idx ');']);
        eval(['L((j+1):end,j' idx ') =  bsxfun(@times, L((j+1):end,j' idx '), den);']);
    end
    eval(['L(d,d' idx ') = real(sqrt( A(d,d' idx ') - multiprod( ' ...
        'L(d,1:(d-1)' idx ') , multitrans ( L(d,1:(d-1)' idx ') ) )) );']);
    clear A;
    L = multitrans(L);
    
    return
    
elseif strcmp(type,'matlab')
    M=size(A,3);
    L = nan(size(A));
    for k=1:M
        L(:,:,k)=chol(A(:,:,k));
    end
    
elseif strcmp(type,'robust')
    M=size(A,3);
    L = nan(size(A));
    for k=1:M
        A_temp = A(:,:,k);
        [V,D] = eig(A_temp);        
        D=diag(D);
        D((D<=0))=eps;
        D=diag(D);
        A_temp = V*D*V';
        L(:,:,k)=chol(A_temp);
    end
    
elseif strcmp(type,'random_sampling')
    if size(A,1) ~= 2
        error('this works only for matrix 2 2');
    end
    L = nan(size(A));
    
    sA=size(A);
    M=prod(sA(3:end));
    A = reshape(A,[2 2 M]);
    
    detA = squeeze(A(1,1,:).*A(2,2,:)-A(1,2,:).*A(2,1,:));
    trA = squeeze(A(1,1,:) + A(2,2,:));
    corr = abs(squeeze(A(1,2,:) ./ sqrt(abs(A(1,1,:).*A(2,2,:))))-1);
    nrj = squeeze(sum(sum(A.^2,2),1));
    
%     idx = (nrj < eps) ... % negligeable or null : often non invertible
%         |(corr < eps) ... % non invertible : too much correlation
    idx =  (corr < eps) ... % non invertible : too much correlation
        | ( detA <= 0 ) ... % one negative or null eigenvalue
        | ( (detA > 0) & trA < 0 ); % two negative eigenvalue
    if any(idx)
        A_temp(1,1,:)=A(1,1,idx);
        A_temp(2,1,:)=A(2,2,idx);
        sA_temp = size(A_temp,3);
        A_temp=A_temp(:);
        A_temp((A_temp<0))=0;
%         A_temp = reshape(A_temp,[2 1 sA_temp]);
%         L(:,1,idx)=sqrt(A_temp);
%         L(:,2,idx)=0;
        A_temp = reshape(A_temp,[1 2 sA_temp]);
        L(1,:,idx)=sqrt(A_temp);
        L(2,:,idx)=0;
    end
    
    L(:,:,~idx)=multi_chol(A(:,:,~idx));
    
    L = reshape(L,[2 2 sA(3:end)]);

%     for k=1:M
%         A_temp = A(:,:,k);
%         deta=det(A_temp);
% %         [V,D] = eig(A_temp);
%         
% %         D=diag(D);
% %         idx = (D<=eps);
%         if (cond(A_temp) > 1/eps) ... % non invertible
%             || ( deta < 0 ) ... % one negative eigenvalue
%             || ( (deta > 0) && trace(A_temp) < 0 ) % two negative eigenvalue
% %         if any(idx)
%             A_temp=diag(A_temp);
%             A_temp((A_temp<0))=0;
%             L(:,1,k)=sqrt(A_temp);
%             L(:,2,k)=0;
%         else
% %             D=diag(D);
% %             A_temp = V*D*V';
%             L(:,:,k)=chol(A_temp);
%         end
% 
%     end
    
else
    error('wrong type');
    
end