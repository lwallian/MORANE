function unitTestCovs()

covMatrix = zeros(1000,1000);

for i = 1:1000
    for j = 1:1000
        covMatrix(i,j) = log( abs(sin((i-1)*(j-1))*cos((i-1)+(j-1))) + 1);
    end
end

cov_s = estimateCovS(covMatrix);
save('cov_s','cov_s','-v7');

deriv_cov_s = diff(cov_s);
deriv_cov_s = [ 0; deriv_cov_s ]; 
deriv_cov_s = deriv_cov_s/0.05;
save('deriv_cov_s','deriv_cov_s','-v7')

tau = sqrt(2 * mean(cov_s.^2) / mean( deriv_cov_s.^2));

tau

end

