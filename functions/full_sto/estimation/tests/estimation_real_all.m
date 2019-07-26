function result = estimation_real_all(param_ref,theta_dBt,alpha_dBt,bool_theta,bool_alpha)
dt = param_ref.dt;
N_tot = param_ref.N_tot;
%     N_tot = param_ref.N_tot;
T = dt*N_tot;
m = param_ref.nb_modes;
clear param_ref;

% The last time step is not used
N_tot = N_tot - 1;
T = T -dt;
alpha_dBt(:,:,end)=[];
theta_dBt(end,:)=[];

%keyboard;
if (bool_theta == true && bool_alpha == false)
    theta_theta = theta_dBt'*theta_dBt;
    theta_theta = theta_theta/T;
    result = theta_theta;
    clear theta_theta;
    
elseif (bool_theta == false && bool_alpha == true)
    alpha_dBt = alpha_dBt(2:end,:,:);
    alpha_dBt = reshape(alpha_dBt,[m^2,N_tot]);
    alpha_alpha = alpha_dBt*alpha_dBt';
    alpha_alpha = alpha_alpha/T;
    result = alpha_alpha;
    clear alpha_alpha;
    
else
    % compute (theta_i + alpha_0i)dBt and alpha_dBt
    alpha_0i = alpha_dBt(1,:,:);
    alpha_0i = permute(alpha_0i,[3 2 1]);
    theta_alpha_0i = theta_dBt + alpha_0i; %(N_tot,m)
    clear theta_dBt alpha_0i;
    theta_alpha_0i = theta_alpha_0i';%(m,N_tot)
%     theta_alpha_0i = permute(theta_alpha_0i,[2 1]);%(m,N_tot)
    
    alpha_dBt = alpha_dBt(2:end,:,:);
    alpha_dBt = reshape(alpha_dBt,[m^2 N_tot ]);%(m^2,N_tot)
    
    
    %% compute the real estimation
    % compute theta_theta_real
    theta_theta_real = (theta_alpha_0i*theta_alpha_0i')/T;
    
    % compute alpha_theta_real
    alpha_theta_real = (alpha_dBt*theta_alpha_0i')/T;
    % compute alpha_alpha_real
    alpha_alpha_real = (alpha_dBt*alpha_dBt')/T;
    clear alpha_dBt;
    
    % computer result
    result_1 = [theta_theta_real;alpha_theta_real];
    result_2 = [alpha_theta_real';alpha_alpha_real];
    result = [result_1,result_2];
end



end