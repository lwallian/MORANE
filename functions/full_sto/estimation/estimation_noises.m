function [result,pseudo_chol] = estimation_noises(param,bt)
% This function estimates the covariance of the additive and multiplicative
% noises, assuming that the Chronos are orthogonal


%% Get parameters
param = fct_name_file_noise_cov(param);

if exist(param.name_file_noise_cov, 'file')==2
    load(param.name_file_noise_cov, 'pseudo_chol');
    result = nan;
else
    M = param.M;
    dX = param.dX;
    MX = param.MX;
    d = param.d;
    m = param.nb_modes;
    N_tot = param.N_tot;
    dt = param.dt;
    T = dt * N_tot;
    lambda = param.lambda;
    param.replication_data = false;
    
    % The last time step is not used
    T = T - dt;
    
    d_bt = bt(2 : end, :) - bt(1 : end - 1, :);
    
    %% compute the RHS of equation
    % R1 for the equation for finding theta_theta
    % R2 for the equation for finding alpha_theta
    % R3 for the equation for finding alpha_alpha
    
    if param.big_data
        [R1, R2, R3] = fct_comp_RHS_big_data(param, bt, d_bt);
    else
        [R1, R2, R3] = fct_comp_RHS(param, bt, d_bt);
    end
    
    % Compute alpha_theta
    alpha_theta = bsxfun(@times, 1 ./ (T * lambda), R2);
    
    % Compute theta_theta
    theta_theta = R1 / T;
    
    % Compute alpha_alpha
    alpha_alpha = bsxfun(@times, 1 ./ (T * lambda), R3);
    
    clear R1 R2 R3
    
    alpha_alpha = reshape(alpha_alpha, [m^2, m^2]);
    alpha_theta = reshape(alpha_theta, [m^2, m]);
    
    result1 = [theta_theta; alpha_theta];
    result2 = [alpha_theta'; alpha_alpha];
    result = [result1, result2];
    clear result1 result2;
    
    %% Force the symetry and the positivity of the matrix
    result = 1/2*(result +result');
    [V, D] = eig(result);
    D = diag(D);
    D(D<0) = 0;
    result = V * diag(D) * V';
    
    pseudo_chol = V*diag(sqrt(D));
    
    % To circumvent the effect of thresholding on the downsampling rate
    if strcmp(param.decor_by_subsampl.choice_n_subsample, 'lms') || ...
            strcmp(param.decor_by_subsampl.choice_n_subsample, 'truncated') || ...
            strcmp(param.decor_by_subsampl.choice_n_subsample, 'htgen')
        pseudo_chol = pseudo_chol * sqrt(param.decor_by_subsampl.tau_corr / param.decor_by_subsampl.n_subsampl_decor);
    end
    
    % %% Remove temporary files
    % rmdir(param.folder_file_U_temp,'s')
    save(param.name_file_noise_cov,'pseudo_chol','-v7.3');
    
end

end