%****************************************************************%
%*****   Test on full stochastic depend on time and space  ******%
%****************************************************************%

% This script test the function estimation_test_all 

clear;
%clc;
disp('***************************************************************')
disp('** In this algorithm, we will simulate all the posible      ***')
disp('** conditions (only theta, only alpha and both alpha theta  ***')
disp('***************************************************************')

bool_theta = true; % simulation without considering theta
bool_alpha = true;% simulation by considering alpha

bool_replic = [false true];

% for n = 2:2:4
% for k = 1:1
for k = 1:2
n=2;
    nb_modes = n
    
    param = fct_def_param(nb_modes);
    
    
    param.replication_data = bool_replic(k);
    
    %***************************
    %%          First step     **
    %***************************
    
    % I compute the value of U' matrix covariance C lambda, phi and the chrono bt
    
    %(1)? compute phi and bt
    [param, bt]=POD_and_POD_knowing_phi(param);
    
    %(2)? Define and save the estimator and U_res
    [param,theta_dBt,alpha_dBt] = fct_estimator_U_res_all(param,bt);
    b_0 = bt(1,:);
    %     b_0 = sum(bt(1:param.N_tot,:))/param.N_tot;
    % b_0(2)=2*b_0(2);
    
%     for i=1:n
%         figure;plot(bt(:,i))
%     end
%     %     clear bt;
    
    %*************************
    %       second step     **
    %*************************
    % I compute the value of the real estimation theta_theta, alpha_theta, and
    % alpha_alpha by using theta_theta_real_ij =
    % sum_tk(theta_dBt_i)*(theta_dBt_j)/T
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    %*************************
    %%    third step         **
    %*************************
    
    if ~(bool_theta || bool_alpha)== true
        error('Your test does not work, please check your boolean again!');end
    
    % compute the real estimation
    result_real = estimation_real_all(param,theta_dBt,alpha_dBt,bool_theta,bool_alpha);
    
    % Simulati~on on d_bt and bt
    %   [bt_new] = fct_POD_bt_dbt_without_theta_1(param,b_0);
    bt_new = fct_POD_bt_dbt_all(param,theta_dBt,alpha_dBt,b_0,bool_theta,bool_alpha);
   
    %*************************
    %% Fourth step           **
    %*************************
    % Compute the estimation test
    result_test = estimation_test_all(param,bt_new,bool_theta,bool_alpha);
    result_test=result_test*result_test';
    
    %% compute the relative error
%     rel_error = abs((result_real - result_test)./result_real)
% %     keyboard;
% 
% %     result_test = reshape(result_test,[n n n n]);
% %     for i=1:n
% %         result_test(i,i,:,:)=0;
% %         result_test(:,:,i,i)=0;
% %         for j=(i+1):n
% %             result_test(i,j,:,:) = ...
% %                 1/2*(result_test(i,j,:,:)-result_test(j,i,:,:));
% %             result_test(j,i,:,:) = - result_test(i,j,:,:);
% %         end    
% %         for j=(i+1):n
% %             result_test(:,:,i,j) = ...
% %                 1/2*(result_test(:,:,i,j)-result_test(:,:,j,i));
% %             result_test(:,:,j,i) = - result_test(:,:,i,j);
% %         end        
% %     end
% %     result_test = reshape(result_test,[n^2 n^2]);
% %     
% %     
% %     rel_error = abs((result_real - result_test)./result_real)
%     
%     
%     % arrange matrix result_est to matrix positive definie
%     result_test = 1/2*(result_test +result_test');
%     [V,D]=eig(result_test);
%     D=diag(D);
%     D(D<0)=0;
%     result_test=V*diag(D)*V';
    
    rel_error = abs((result_real - result_test)./result_real)
    
    
%     %****************************************
%     % check the characteristic time_scale  **
%     %****************************************
%     T = param.N_tot*param.dt
%     tau = result_real;
%     tau = trace(tau);
%     tau = abs(1/tau)
%     %     dt = tor/10000;
%     %     T = 10*tor
%     %     N = round(T/dt)
    keyboard;
end