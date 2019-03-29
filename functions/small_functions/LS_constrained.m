function beta=LS_constrained(X,Y,beta_min)

if nargin == 2
    beta_min=-inf;
end
idx=[];
bool=true;
n_ref=size(X,2);
n=n_ref;
list_ref=1:n_ref;

while bool && n>0
    beta_temp=LS(X,Y);
    
    idxtest=beta_temp<beta_min;
    bool=any(idxtest);
    if bool
%         beta=nan(n_ref,1);
%         beta(idx)=beta_min*ones(n_ref-n,1);
%         [~, idx_temp] = min(beta);
        [~, idx_temp] = min(beta_temp);
%         idx=[idx idx_temp];
        list_ref(idx_temp)=[];
        Y=Y-X(:,idx_temp)*beta_min;
        X(:,idx_temp)=[];
        n=size(X,2);
    end
end
% idx=sort(idx);
beta=nan(n_ref,1);
beta(list_ref)=beta_temp;
beta(isnan(beta))=beta_min*ones(n_ref-n,1);
% beta(idx)=beta_min*ones(n_ref-n,1);
% if n>0
%     beta(isnan(beta))= beta_temp;
% end

    function beta=LS(X,Y)
        if size(X,1)>n
            beta = (X'*X)\(X'*Y);
%         beta= (X'*X)\X'*Y;
        else
            beta = (X'/(X*X'))*Y;
        end
    end

end