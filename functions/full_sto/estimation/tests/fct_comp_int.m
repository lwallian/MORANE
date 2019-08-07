function int_b = fct_comp_int(param,bt)

    dt = param.dt;
%     m = param.nb_modes;
%     N_tot = param.N_tot;

    int_b = sum(bt(1:end-1,:))*dt;
    
% % load bt
%    % load(param.name_file_bt);
%     
% % compute int_b
%      int_b = zeros(m,1);
%      for k = 1:m
%         for t = 1:N_tot-1
%             f = (bt(t,k)+bt(t+1,k))/2;
%             int_b(k,1) = int_b(k,1) +f*dt;
%         end
%      end
% end

