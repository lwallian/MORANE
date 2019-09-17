function [result, pseudo_chol, eta_0, Mi_ss_0] = estimation_correlated_noises(param, bt)
% This function estimates the covariance of the additive and multiplicative
% noises, assuming that the Chronos are orthogonal


%% Get parameters
param = fct_name_file_correlated_noise_cov(param);

if exist(param.name_file_noise_cov,'file')==2
    load(param.name_file_noise_cov,'pseudo_chol', 'Mi_sigma', 'eta_0', 'Mi_ss_0');
    result = nan;
else
    m = param.nb_modes;
    
    %% compute the RHS of equation
    [theta_theta, xi_xi_inf, eta_0, Mi_ss_0] = fct_comp_correlated_RHS(param, bt);
    
    theta_theta = reshape(theta_theta, [m * (m + 1), m * (m + 1)]);
    xi_xi_inf = reshape(xi_xi_inf, [m, m]);
    theta_xi = zeros(m * (m + 1), m);
    % Mi_sigma is an array
    
    result1 = [xi_xi_inf; theta_xi];
    result2 = [theta_xi'; theta_theta];
    result = [result1, result2];
    clear result1 result2;
    
    %% Force the symetry and the positivity of the matrix
    result = 1 / 2 * (result + result');
    [V, D] = eig(result);
    D = diag(D);
    D(D < 0) = 0;
    result = V * diag(D) * V';
    
    
    pseudo_chol = V * diag(sqrt(D));
    
    % %% Remove temporary files
    % rmdir(param.folder_file_U_temp,'s')
    save(param.name_file_noise_cov, 'pseudo_chol', 'eta_0', 'Mi_ss_0','-v7.3');
    
end

end

