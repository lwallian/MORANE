function [result, pseudo_chol] = estimate_noise_non_orthogonal(bt, eta, n, T, dt, n_particles, diagonal_G, lambda)
%ESTIMATE_NOISE_NON_ORTHOGONAL Test function for the noise estimator in the
%correlated case. It uses the hypothesis of non orthogonal chronos.
%   @param bt: chronos
%   @param eta: multiplicative noise
%   @param n: number of modes
%   @param T: number of time steps
%   @param dt: time step delta
%   @param n_particles: number of particles to evolve
%   @param diagonal_G: if true, it uses the lambda to generate a diagonal G
%   matrix with those values and a 1 for b0.
%   @param lambda: lambda values for when diagonal_G is true, otherwise it
%   should not have a value
%   @return result: noise matrix without the pchol
%   @return pseudo_chol: pseudo cholesky on the result matrix

diff_bt = (bt(2 : end, :, :) - bt(1 : end - 1, :, :)) / (dt);
diff_bt(end, :, :, :) = [];
eta(1, :, :, :) = [];
eta(end, :, :, :) = [];
bt(1, :, :) = [];
bt(end, :, :) = [];
T = T - 2;
d2bt = (diff_bt(2 : end, :, :) - diff_bt(1 : end - 1, :, :)) / (dt);
d2bt_c = d2bt - mean(d2bt, 1);
deta = (eta(2 : end, :, :) - eta(1 : end - 1, :, :)) / (dt);
deta_c = deta - mean(deta, 1);
bt_x = cat(2, bt, ones(T, 1, n_particles));

if diagonal_G
    G_pq = diag(lambda);
else
    G_pq = zeros(n + 1, n + 1, n_particles);
    for k = 1 : T
        for p = 1 : n + 1
            for q = 1 : n + 1
                G_pq(p, q, :) = G_pq(p, q, :) + bt_x(k, p, :) .* bt_x(k, q, :);
            end
        end
    end
    G_pq = G_pq ./ T;
end

% Beta
beta = zeros(n + 1, n, n + 1, n, n_particles);
for l = 1 : n_particles
    for p = 1 : n + 1
        for i = 1 : n
            for q = 1 : n + 1
                for j = 1 : n
                    for k = 1 : T - 1
                        beta(p, i, q, j, l) = beta(p, i, q, j, l) + ...
                            bt_x(k, p, l) * d2bt_c(k, i, l) * deta_c(k, q, j, l);
                    end
                end
            end
        end
    end
end
beta = beta .* dt ./ T;

% theta_theta_est
theta_theta_est = zeros(n + 1, n, n + 1, n, n_particles);
for k = 1 : n_particles
    weights_inv = diag(sqrt(diag(G_pq(:, :, k))).^(-1));
    weighted_G_pq = weights_inv * G_pq(:, :, k) * weights_inv;
    for i = 1 : n
        for q = 1 : n + 1
            for j = 1 : n
%                 weights_inv = diag(sqrt(diag(G_pq(:, :, k))).^(-1));
%                 weighted_G_pq = weights_inv * G_pq(:, :, k) * weights_inv;
                weighted_beta = weights_inv * beta(:, i, q, j, k);
                theta_theta_est(:, i, q, j, k) = weighted_G_pq \ weighted_beta;
                theta_theta_est(:, i, q, j, k) = weights_inv * theta_theta_est(:, i, q, j, k);
            end
        end
    end
end

% xi_xi_est
xi_xi_est = zeros([n, n, n_particles]);
for k = 1 : n_particles
    for i = 1 : n
        for j = 1 : n
            kappa = d2bt_c(:, i, k)' * d2bt_c(:, j, k) * dt / T;
            gamma = 0;
            for p = 1 : n + 1
                for q = 1 : n + 1
                    gamma = gamma + G_pq(p, q, k) * theta_theta_est(p, i, q, j, k);
                end
            end
            xi_xi_est(i, j, k) = kappa - gamma;
        end
    end
end

theta_theta_est = reshape(theta_theta_est, [n * (n + 1), n * (n + 1)]);
xi_xi_est = reshape(xi_xi_est, [n, n]);
theta_xi = zeros(n * (n + 1), n);

result1 = [xi_xi_est; theta_xi];
result2 = [theta_xi'; theta_theta_est];
result = [result1, result2];
clear result1 result2;

% Force the symetry and the positivity of the matrix
result = 1 / 2 * (result + result');
[V, D] = eig(result);
D = diag(D);
D(D < 0) = 0;
result = V * diag(D) * V';
pseudo_chol = V * diag(sqrt(D));

end

