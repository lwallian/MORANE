function Q = operator_Q(xi, phi, param)

m = param.nb_modes;
nu = param.viscosity;
M = param.M;
dX = param.dX;
MX = param.MX;
d = param.d;

Q = zeros(m + 1, m);

% Estimate the xi's gradient
xi = permute(xi, [3, 4, 1, 2]);%(1,d,M)
xi = reshape(xi, [1, d, MX]);%(1,d,Mx,My,(Mz))

dxi = gradient_mat(xi, dX);
dxi = permute(dxi, [ndims(dxi) + 1, 1, ndims(dxi), 3 : ndims(dxi) - 1, 2]);
xi = permute(xi, [ndims(xi) + 1, 1 : ndims(xi)]);%(1,1,d,Mx ,My,(Mz))

% Do the projection for each q and i
for q = 1 : m + 1
    % Small scale advected by large scale
    phi_q = phi(:, q, :);%(M,1,d)
    phi_q = permute(phi_q, [2, 3, 1]);%(1,d,M)
    phi_q = reshape(phi_q, [1, d, MX]);%(1,d,Mx,My,(Mz))
    dphi_q = gradient_mat(phi_q, dX);
    dphi_q = permute(dphi_q, [ndims(dphi_q) + 1, 1, ndims(dphi_q), 3 : ndims(dphi_q) - 1, 2]);
    %(1,1,d!,Mx,My,(Mz),d
    
    adv_sl = bsxfun(@times, dphi_q, xi); % 1 x (1) x d x Mx x My (x Mz) d
    clear dphi_q;
    adv_sl = sum(adv_sl, 3);
    adv_sl = permute(adv_sl, [1, 2, 4 : ndims(adv_sl), 3]);%(1 1 Mx My (Mz) d)
    
    % Large scale advected by small scale
    phi_q = permute(phi_q, [ndims(phi_q) + 1, 1 : ndims(phi_q)]);
    adv_ls = bsxfun(@times, dxi, phi_q);
    clear psi_q;
    adv_ls = sum(adv_ls, 3);
    adv_ls = permute(adv_ls, [1, 2, 4 : ndims(adv_ls), 3]);%(1 1 Mx My (Mz) d)
    
    % Do the divergence free projection
    integ = adv_sl + adv_ls;
    integ = permute(integ, [3 : ndims(integ) - 1, 1, 2, ndims(integ)]); % [Mx, My, (Mz), 1, 1, d]
    integ = reshape(integ, [M, 1, d]);
    if strcmp(param.type_data, 'turb2D_blocks_truncated')
        integ = integ - proj_div_propre(integ, MX, dX, true);
    else
        integ = integ - proj_div_propre(integ, MX, dX, false);
    end
    integ = reshape(integ, [MX, 1, d]);
    integ = permute(integ, [ndims(integ) - 1, 1 : ndims(integ) - 2, ndims(integ)]);
    integ = reshape(integ, [1, 1, MX, d]);
    
    % Add the diffusion term if phi_0
    if q == m + 1
        xi = reshape(xi, [1, d, MX]);
        Lap_xi = laplacian_mat(xi, dX);
        Lap_xi = nu*Lap_xi;
        Lap_xi = permute(Lap_xi,[1 ndims(Lap_xi)+1 2:ndims(Lap_xi)]);
        Lap_xi = permute(Lap_xi, [1 2 4:ndims(Lap_xi) 3]);%(1,1,Mx,My,(Mz),d)
        xi = permute(xi, [ndims(xi) + 1, 1 : ndims(xi)]);%(1,1,d,Mx ,My,(Mz))
        
        integ = integ + Lap_xi;
    end
    
    % projection on phi_j
    for j = 1 : m
        phi_j = phi(:, j, :);
        phi_j = permute(phi_j, [4, 2, 1, 3]);%(1,1,M,d)
        phi_j = reshape(phi_j, [1, 1, MX, d]);%(1,1,Mx,My,(Mz),d)
        
        s_temp = integ .* phi_j; %(1,1,Mx,My,(Mz),d)
        clear psi_j;
        s_temp = sum(s_temp, ndims(s_temp));%(1,1,Mx,My,(Mz))
        
        % compute the integration of s_temp
        s_temp = integration_mat(s_temp, dX);
        Q(q,j) = -s_temp;
        
        clear s_temp;
    end
    clear adv_sl adv_ls Lap_xi;
end

end

