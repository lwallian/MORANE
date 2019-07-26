function bt = fct_POD_bt_dbt_all(param_ref,theta_dBt,alpha_dBt,bt_0,bool_theta,bool_alpha)

N_tot = param_ref.N_tot;
N_tot = N_tot - 1;
%dt = param_ref.dt;
m = param_ref.nb_modes;

d_bt = zeros(N_tot,m);
bt= [bt_0;zeros(N_tot,m)];
%keyboard;
%% compute d_bt and bt without using alpha_dBt
if (bool_theta == true && bool_alpha== false)
    % compute d_bt and bt
    for t = 1:N_tot
        d_bt(t,:) = theta_dBt(t,:);
        bt(t+1,:) = bt(t,:) + d_bt(t,:);
    end
    
%% compute d_bt and bt without using theta_dBt
elseif (bool_theta == false && bool_alpha== true)
    alpha_dBt = alpha_dBt(2:end,:,:);
    % compute d_bt and bt
    for t = 1:N_tot
        for i = 1:m
            al_temp = alpha_dBt(:,i,t);
            d_bt(t,i) = d_bt(t,i) + al_temp' * bt(t,:)';
            bt(t+1,i) = bt(t,i) + d_bt(t,i);
            clear bt_temp;
            clear al_temp;
        end
    end
    
%% compute d_bt and bt by using theta_dBt and alpha_dBt
else
    % compute d_bt and bt
    for t = 1:N_tot
        for i = 1:m
            bt_temp = [1 bt(t,:)];
            al_temp = alpha_dBt(:,i,t);
            d_bt(t,i) = d_bt(t,i) + al_temp'*bt_temp' + theta_dBt(t,i);
            bt(t+1,i) = bt(t,i) + d_bt(t,i);
            clear bt_temp;
            clear al_temp;
        end
    end
end

end
