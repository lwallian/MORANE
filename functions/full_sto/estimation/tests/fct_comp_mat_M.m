function mat_M = fct_comp_mat_M(param_ref,bt)

dt = param_ref.dt;
% N_tot = param_ref.N_tot;
% m = param_ref.nb_modes;
% %keyboard
%     b_0 = ones(N_tot,1);
%     bt = [b_0,bt];%(N_tot,m+1)
%     clear b_0;

% compute matrix M :: M_kp = int(b_k*b_p)

mat_M = bt(1:end-1,:)'*bt(1:end-1,:)*dt;
%   mat_M = zeros(m+1,m+1);
%     for p = 1:m+1
%         for k = 1:m+1
%             for t = 1:N_tot
%                 mat_M(k,p) = mat_M(k,p)+bt(t,p)*bt(t,k)*dt;
%             end
%         end
%     end

fprintf(['the conditionning number of mat_M is ' num2str(cond(mat_M)) ' \n']);
if any(isnan(mat_M)) error('fsaas');end

%keyboard;
end
