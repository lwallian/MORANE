%****************************************************************%
% This script test verify that the functions estimation_noises and
% estimation_test_all compute the same things when the Chonos are orthogonal

bool_replic = [false true];

n=2;
nb_modes = n

param = fct_def_param(nb_modes);

%(1)? compute phi and bt
[param, bt]=POD_and_POD_knowing_phi(param);

% Compute the reference estimation
param.replication_data=false;
result_test = estimation_test_all(param,bt,true,true);

% Compute the estimation test
result_real = estimation_noises(param,bt);

%% compute the relative error
result_test
rel_error = abs((result_real - result_test)./result_real)
%     keyboard;

%     result_test = reshape(result_test,[n n n n]);
%     for i=1:n
%         result_test(i,i,:,:)=0;
%         result_test(:,:,i,i)=0;
%         for j=(i+1):n
%             result_test(i,j,:,:) = ...
%                 1/2*(result_test(i,j,:,:)-result_test(j,i,:,:));
%             result_test(j,i,:,:) = - result_test(i,j,:,:);
%         end
%         for j=(i+1):n
%             result_test(:,:,i,j) = ...
%                 1/2*(result_test(:,:,i,j)-result_test(:,:,j,i));
%             result_test(:,:,j,i) = - result_test(:,:,i,j);
%         end
%     end
%     result_test = reshape(result_test,[n^2 n^2]);
%
%
%     rel_error = abs((result_real - result_test)./result_real)


% arrange matrix result_est to matrix positive definie
result_test = 1/2*(result_test +result_test');
[V,D]=eig(result_test);
D=diag(D);
D(D<0)=0;
result_test=V*diag(D)*V';

rel_error = abs((result_real - result_test)./result_real)

keyboard;
