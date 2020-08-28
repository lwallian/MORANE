function test_reductionMatrix()

param.nb_modes = 2;
m = param.nb_modes;
lambda = zeros(param.nb_modes,1);
param.svd_pchol = 2;
param.correlated_model = false;
 
for i = 1:param.nb_modes
    lambda(i) = -i^2 + i + 100;
end

param.lambda = lambda/500;

pchol_cov_noises = zeros(m*(m+1),m*(m+1));

for i = 1:(m*(m+1))
    for j = 1:(m*(m+1))
        pchol_cov_noises(i,j) = i*j*sin(i*j) + i*j - cos(j) - cos(i);
    end
end

pchol_cov_noises

pchol_cov_noises = reductionMatrix(param,pchol_cov_noises);

pchol_cov_noises

end


function [pchol_cov_noises] = reductionMatrix(param,pchol_cov_noises)

correlated_model = param.correlated_model;

if param.svd_pchol>0
%     figure;plot(diag(pchol_cov_noises*pchol_cov_noises'))
%     cov = pchol_cov_noises*pchol_cov_noises';
    
    % Normalizing noise covariance
    sq_lambda = sqrt(param.lambda);
%     sq_lambda = ones(size(sq_lambda));
    if correlated_model
%         sq_lambda_cat = [ sq_lambda' 1 ];
%     else
%         sq_lambda_cat = [ sq_lambda' ];
        sq_lambda_cat = [ 1 sq_lambda' 1 ];
    else
        sq_lambda_cat = [ 1 sq_lambda' ];
    end
    pchol_add_noise = pchol_cov_noises(1:param.nb_modes,:);
    pchol_cov_noises(1:param.nb_modes,:)=[];
    n_plus = length(sq_lambda_cat);
    pchol_cov_noises = reshape( pchol_cov_noises , ...
        [ (n_plus-1) param.nb_modes n_plus*param.nb_modes ] );
    
%     for q=1:(n_plus*param.nb_modes)
%         pchol_cov_noises(1:end,:,q)
%     end
    
    %     if ~correlated_model
    pchol_add_noise = permute( pchol_add_noise ,[ 3 1 2]);
    pchol_cov_noises = cat(1,pchol_add_noise,pchol_cov_noises);
    %     end
    
    for k=1:n_plus
        switch  param.svd_pchol
            case 1
                for i=1:(param.nb_modes)
                    %             (sq_lambda_cat(k)/sq_lambda(i))
                    pchol_cov_noises(k,i,:) = (sq_lambda_cat(k)/sq_lambda(i)) * ...
                        pchol_cov_noises(k,i,:);
                end
            case 2
                pchol_cov_noises(k,:,:) = (sq_lambda_cat(k)) * ...
                    pchol_cov_noises(k,:,:);
        end
    end
    
    %     if ~correlated_model
    pchol_add_noise = pchol_cov_noises(1,:,:);
    pchol_cov_noises(1,:,:)=[];
    %     end
    
%     for q=1:(n_plus*param.nb_modes)
%         pchol_cov_noises(1:end,:,q)
%     end
    
    pchol_cov_noises = reshape( pchol_cov_noises , ...
        [ (n_plus-1)*param.nb_modes n_plus*param.nb_modes ] );
%     pchol_cov_noises = cat(1,pchol_add_noise,pchol_cov_noises);
    
    %     if ~correlated_model
    pchol_add_noise = permute( pchol_add_noise ,[ 2 3 1]);
    pchol_cov_noises = cat(1,pchol_add_noise,pchol_cov_noises);
    %     end
    
%     
%     figure;plot(diag(pchol_cov_noises*pchol_cov_noises'))
%     cov2 = pchol_cov_noises*pchol_cov_noises';
%     figure;imagesc(cov2)
%     cov-cov2
    
%     cov2 = cov;
    
    % Noise PCA
    var_pchol_cov_noises_ini = trace(pchol_cov_noises*pchol_cov_noises');
    [U_cov_noises,S_cov_noises,~] = ...
        svds(pchol_cov_noises,param.nb_modes);
    pchol_cov_noises = U_cov_noises * S_cov_noises;
    
    var_pchol_cov_noises_red = trace(pchol_cov_noises*pchol_cov_noises');
    ratio_var_red_pchol = var_pchol_cov_noises_red/var_pchol_cov_noises_ini

    
%     figure;plot(diag(pchol_cov_noises*pchol_cov_noises'))
%     cov2 = pchol_cov_noises*pchol_cov_noises';
%     figure;imagesc(cov2)
    
    % de-Normalizing noise covariance
    pchol_add_noise = pchol_cov_noises(1:param.nb_modes,:);
    pchol_cov_noises(1:param.nb_modes,:)=[];
    
    pchol_cov_noises = reshape( pchol_cov_noises , ...
        [ (n_plus-1) param.nb_modes param.nb_modes ] );
    
%     for q=1:(n_plus*param.nb_modes)
%         pchol_cov_noises(1:end,:,q)
%     end
    
    %     if ~correlated_model
    pchol_add_noise = permute( pchol_add_noise ,[ 3 1 2]);
    pchol_cov_noises = cat(1,pchol_add_noise,pchol_cov_noises);
    %     end
    
    
    for k=1:n_plus
        switch  param.svd_pchol
            case 1
                for i=1:(param.nb_modes)
                    pchol_cov_noises(k,i,:) = (sq_lambda(i)/sq_lambda_cat(k)) * ...
                        pchol_cov_noises(k,i,:);
                end
            case 2
                pchol_cov_noises(k,:,:) = (1/sq_lambda_cat(k)) * ...
                    pchol_cov_noises(k,:,:);
        end
    end
        
    %     if ~correlated_model
    pchol_add_noise = pchol_cov_noises(1,:,:);
    pchol_cov_noises(1,:,:)=[];
    %     end
    
%     for q=1:(n_plus*param.nb_modes)
%         pchol_cov_noises(1:end,:,q)
%     end
    
    pchol_cov_noises = reshape( pchol_cov_noises , ...
        [ (n_plus-1)*param.nb_modes param.nb_modes ] );
%     pchol_cov_noises = cat(1,pchol_add_noise,pchol_cov_noises);
    
    %     if ~correlated_model
    pchol_add_noise = permute( pchol_add_noise ,[ 2 3 1]);
    pchol_cov_noises = cat(1,pchol_add_noise,pchol_cov_noises);
    %     end
    
    
%     figure;plot(diag(pchol_cov_noises*pchol_cov_noises'))
%     
%     figure;imagesc(cov);cax=caxis; colorbar;
%     cov2 = pchol_cov_noises*pchol_cov_noises';
%     figure;imagesc(cov2);caxis(cax); colorbar;
    
end

end

