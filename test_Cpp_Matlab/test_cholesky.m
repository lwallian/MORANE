function test_cholesky()

nModesU = 2;
size_pchol = nModesU*(nModesU+1);

result = zeros(size_pchol,size_pchol);

for i = 1:size(result,1)
    for j = 1:size(result,2)
        result(i,j) = cos((i-1)*j);
    end
end

result

result = 1/2*(result +result');

result
[V, D] = eig(result);
D = diag(D);

V*diag(sqrt(D))*diag(sqrt(D))*V'

D
D(D<0) = 0;
    V

diag(sqrt(D))
pseudo_chol = V*diag(sqrt(D));

V*diag(sqrt(D))*diag(sqrt(D))*V'

pseudo_chol

end

