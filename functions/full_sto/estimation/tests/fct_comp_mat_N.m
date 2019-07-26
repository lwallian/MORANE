function mat_N = fct_comp_mat_N(param_ref,MM,int_b)
%% compute matrix N :: N_kp = M_kp - int_b_k
    m = param_ref.nb_modes;
    dt = param_ref.dt;
    N_tot = param_ref.N_tot;
    N_tot = N_tot -1;
    T= dt*N_tot;
    %int_b = comp_int(param_ref);
    
    mat_N = MM * T - int_b' * int_b;
    
%     mat_N = zeros(m,m);
%     for p = 1:m
%         for k = 1:m
%             mat_N(k,p) = T*MM(k,p) - int_b(k)*int_b(p);
%         end
%     end
    fprintf(['the conditionning number of mat_N is ' num2str(cond(mat_N)) ' \n']);
    if any(isnan(mat_N)) error('fsaas');end

end