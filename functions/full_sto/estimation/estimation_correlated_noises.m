function [result, pseudo_chol, Mi_sigma, eta_0, Mi_ss_0] = estimation_correlated_noises(param, bt)
% This function estimates the covariance of the additive and multiplicative
% noises, assuming that the Chronos are orthogonal


%% Get parameters
param = fct_name_file_correlated_noise_cov(param);

if exist(param.name_file_noise_cov,'file')==2
    load(param.name_file_noise_cov,'pseudo_chol', 'Mi_sigma', 'eta_0', 'Mi_ss_0');
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
    
    % The last two time steps are not used
    T = T - 2 * dt;
    
    dbt = bt(2:end,:) - bt(1:end-1,:);
    d2bt = dbt(2:end,:) - dbt(1:end-1,:);
    
    %% compute the RHS of equation
    % R1 is proportional to theta_theta
    % R2 is proportional to Mi_sigma
    % R3 is proportional to xi_xi_inf
    
    [R1, R2, R3, eta_0, Mi_ss_0] = fct_comp_correlated_RHS(param, bt, d2bt);
    % To test that the estimation formulas give a reasonable estimation of
    % the noise terms (uncomment to test)
%     sigma_ss = 1e8 * generate_sigma_ss(0.1, [MX, d], 100, dX);
%     [R1t, R2t, R3t] = fct_general_correlated_RHS(param, bt, d2bt, sigma_ss);
    
    % Compute theta_theta
    theta_theta = bsxfun(@times, 1 ./ (N_tot), R1);
    
    % Compute Mi_sigma
    Mi_sigma = R2 ./ N_tot;
    
    % Compute xi_xi_inf
    xi_xi_inf = R3 ./ N_tot;
    
    clear R1 R2 R3;
    
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
    save(param.name_file_noise_cov, 'pseudo_chol', 'Mi_sigma', 'eta_0', 'Mi_ss_0','-v7.3');
    
end

end

