
function [param,c] = comp_mU_U_temp_matrix_c(param_ref)
%keyboard;
    % Load data
    [U,param_temp]=read_data(param_ref.type_data,param_ref.folder_data);
	[M,N_tot,d] = size(U);
    param_temp = rmfield(param_temp, 'folder_data');
    param_temp = rmfield(param_temp, 'type_data');
    
    param = mergestruct(param_temp,param_ref);
   % param.dt = 0.1;
    clear param_temp;
    clear param_ref;

    % compute m_U
    	m_U = mean(U,2);

    % save m_U
	param.name_file_m_U = [param.folder_data param.type_data '_m_U'];
	save(param.name_file_m_U,'m_U','-v7.3');

    % compute U'
    	U_temp=bsxfun(@minus,U,m_U);
    	clear U;
    	U = U_temp;
    	clear U_temp;

    % save U'
	 param.name_file_U_t_test = [param.folder_data param.type_data '_U_test'];
	 save(param.name_file_U_t_test,'U','-v7.3');

    % compute C
	c = zeros(N_tot);
	if N_tot > M
    		warning('The computing of the mode phi and the coefficients bt should use the fact that N > M');
	end
	for i = 1:N_tot
    		for k=1:d % loop on the dimension
        		U1_temp = U(:,i,k);
        			for j=1:N_tot
            				c(i,j)=c(i,j)+U1_temp'*U(:,j,k);
        			end
   		 end
	end
	
end



