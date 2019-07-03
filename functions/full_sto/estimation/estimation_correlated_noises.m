function [result, pseudo_chol] = estimation_correlated_noises(param, bt)
% This function estimates the covariance of the additive and multiplicative
% noises, assuming that the Chronos are orthogonal


%% Get parameters
param = fct_name_file_correlated_noise_cov(param);

if exist(param.name_file_noise_cov,'file')==2
    load(param.name_file_noise_cov,'pseudo_chol');
    result = nan;
else
    M = param.M;
    dX = param.dX;
    MX = param.MX;
    d = param.d;
    m = param.nb_modes;
    N_tot = param.N_tot;
    dt = param.dt;
    T = dt*N_tot;
    lambda = param.lambda;
    param.replication_data=false;
    
    % The last time step is not used
    T = T -dt;
    
    dbt = bt(2:end,:) - bt(1:end-1,:);
    d2bt = dbt(2:end,:) - dbt(1:end-1,:);
    
    %% compute the RHS of equation
    % R1 is proportional to theta_theta
    % R2 is proportional to Mi_sigma
    
    % NO PARAMETER BIG DATA ?
    [R1,R2] = fct_comp_correlated_RHS(param, bt, d2bt);
    
    % Compute theta_theta
    theta_theta = bsxfun(@times, 1 ./ (lambda * T), R1);
    
    % Compute Mi_sigma
    Mi_sigma = R2 ./ T;
    
    clear R1 R2;
    
    % BELOW IS NOT FINISHED YET
    
    theta_theta = reshape(theta_theta, [m^2, m^2]);
    alpha_theta = reshape(alpha_theta,[m^2,m]);
    
    result1 = [theta_theta;alpha_theta];
    result2 = [alpha_theta'; alpha_alpha];
    result = [result1,result2];
    clear result1 result2;
    
    %% Force the symetry and the positivity of the matrix
    result = 1/2*(result +result');
    [V,D]=eig(result);
    D=diag(D);
    D(D<0)=0;
    result=V*diag(D)*V';
    
    pseudo_chol = V*diag(sqrt(D));
    
    % To circumvent the effect of thresholding on the downsampling rate
    if strcmp(param.decor_by_subsampl.choice_n_subsample, 'corr_time')
        pseudo_chol = pseudo_chol * sqrt(param.decor_by_subsampl.tau_corr / param.decor_by_subsampl.n_subsampl_decor);
    end
    
    % %% Remove temporary files
    % rmdir(param.folder_file_U_temp,'s')
    save(param.name_file_noise_cov,'pseudo_chol','-v7.3');
    
end

end

end

