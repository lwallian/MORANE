function R = operator_R(xi, psi, param)

m = param.nb_modes;
M = param.M;
dX = param.dX;
MX = param.MX;
d = param.d;

R = zeros(m + 1, m);

% Estimate the xi's gradient
xi = permute(xi, [3, 4, 1, 2]);%(1,d,M)
xi = reshape(xi, [1, d, MX]);%(1,d,Mx,My,(Mz))

dxi = gradient_mat(xi, dX);
dxi = permute(dxi, [ndims(dxi) + 1, 1, ndims(dxi), 3 : ndims(dxi) - 1, 2]);
xi = permute(xi, [ndims(xi) + 1, 1 : ndims(xi)]);%(1,1,d,Mx ,My,(Mz))

% Do the projection for each q and i
for q = 1 : m + 1
    % Small scale advected by large scale
    psi_q = psi(:, q, :);%(M,1,d)
    psi_q = permute(psi_q, [2, 3, 1]);%(1,d,M)
    psi_q = reshape(psi_q, [1, d, MX]);%(1,d,Mx,My,(Mz))
    dpsi_q = gradient_mat(psi_q, dX);
    dpsi_q = permute(dpsi_q, [ndims(dpsi_q) + 1, 1, ndims(dpsi_q), 3 : ndims(dpsi_q) - 1, 2]);
    %(1,1,d!,Mx,My,(Mz),d
    
    adv_sl = bsxfun(@times, dpsi_q, xi); % 1 x (1) x d x Mx x My (x Mz) d
    clear dphi_q;
    adv_sl = sum(adv_sl, 3);
    adv_sl = permute(adv_sl, [1, 2, 4 : ndims(adv_sl), 3]);%(1 1 Mx My (Mz) d)
    
    % Large scale advected by small scale
    psi_q = permute(psi_q, [ndims(psi_q) + 1, 1 : ndims(psi_q)]);
    adv_ls = bsxfun(@times, dxi, psi_q);
    clear psi_q;
    adv_ls = sum(adv_ls, 3);
    adv_ls = permute(adv_ls, [1, 2, 4 : ndims(adv_ls), 3]);%(1 1 Mx My (Mz) d)
    
    % Do the projection onto the divergence free space
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
    
    % projection on phi_j
    for j = 1 : m
        psi_j = psi(:, j, :);
        psi_j = permute(psi_j, [4, 2, 1, 3]);%(1,1,M,d)
        psi_j = reshape(psi_j, [1, 1, MX, d]);%(1,1,Mx,My,(Mz),d)
        
        s_temp = integ .* psi_j; %(1,1,Mx,My,(Mz),d)
        clear psi_j;
        s_temp = sum(s_temp, ndims(s_temp));%(1,1,Mx,My,(Mz))
        
        % compute the integration of s_temp
        s_temp = integration_mat(s_temp, dX);
        R(q,j) = - s_temp;
        
        clear s_temp;
    end
    clear adv_sl adv_ls;
end

end

