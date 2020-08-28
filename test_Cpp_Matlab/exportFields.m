function [param] = exportFields(param, nb_file)

folder = '/home/guillaume.lepape@eurogiciel.fr/HDD/flow_past_cylinder/donnees_parallele/ITHACAoutput/residual_speed_txt2/';

threshold_str = num2str(param.decor_by_subsampl.spectrum_threshold);
iii = (threshold_str=='.');
threshold_str(iii)='_';

MX = param.MX;
dX = param.dX;
M = param.M;
N_tot = param.N_tot;
d = 2;

U_calcul = zeros(M,N_tot,d);

file = 60220;

dependance_on_time_of_a = '_a_cst_';

char_filter = [];

param.folder_file_U_temp = ...
        [ param.folder_data 'folder_file_temp_' param.type_data '_' ...
        num2str(param.nb_modes) '_modes_' ...
        dependance_on_time_of_a char_filter ...
        '_decor_by_subsampl_' num2str(param.decor_by_subsampl.meth) ...
        '_choice_' num2str(param.decor_by_subsampl.choice_n_subsample)  ...
        '_threshold_' threshold_str  ...
        'fct_test_' param.decor_by_subsampl.test_fct ];

mkdir('/home/guillaume.lepape@eurogiciel.fr/HDD/REDLUM_CODE/data/folder_file_temp_test_variance_tensor_2_modes__a_cst__decor_by_subsampl_bt_decor_choice_htgen2_threshold_1fct_test_b');
for k = 1:N_tot
    Ndigits = dec2base(file,10) - '0';
    
    if (Ndigits(end) ~= 0)
        file_format = sprintf('%0.2f',file*10^-2);
    elseif (Ndigits(end-1) == 0)
        file_format = sprintf('%0.0f',file*10^-2);
    else
        file_format = sprintf('%0.1f',file*10^-2);
    end
    
    file_format

    fileID = fopen( [folder file_format],'r' );

    C = textscan(fileID,'%f %f %f %f');
  
    for i = 1:MX(1)
        for j = 1:MX(2)

            x = C{1}(i + MX(1)*(j-1));
            y = C{2}(i + MX(1)*(j-1));
            Ux = C{3}(i + MX(1)*(j-1));
            Uy = C{4}(i + MX(1)*(j-1));
       
            floor_x = abs( floor(x/dX(1)) - (x/dX(1)) );
            floor_y = abs( floor(y/dX(2)) - (y/dX(2)) );
            ceil_x = abs( ceil(x/dX(1)) - (x/dX(1)) );
            ceil_y = abs( ceil(y/dX(2)) - (y/dX(2)) );
        
            if ( (floor_x >= ceil_x) && (floor_y >= ceil_y) )
                ind_x = ceil(x/dX(1)) + 1;
                ind_y = ceil(y/dX(2)) + 1;   
            elseif ( (floor_x >= ceil_x) && (ceil_y >= floor_y) )
                ind_x = ceil(x/dX(1)) + 1;
                ind_y = floor(y/dX(2)) + 1;
            elseif ( (ceil_x >= floor_x) && (floor_y >= ceil_y) )
                ind_x = floor(x/dX(1)) + 1;
                ind_y = ceil(y/dX(2)) + 1;
            elseif ( (ceil_x >= floor_x) && (ceil_y >= floor_y) )
                ind_x = floor(x/dX(1)) + 1;
                ind_y = floor(y/dX(2)) + 1; 
            else
               disp('Full bug al');
            end
       
            if (ind_x > MX(1))
                ind_x = MX(1);
            end
            
            if (ind_y > MX(2))
                ind_y = MX(2);
            end  
    
        
            U_calcul(ind_x + MX(1)*(ind_y-1),k,:) = [Ux Uy];
        end
    end
    fclose(fileID);
    
    file = file + 55;
end
    
x=dX(1)*(0:(MX(1)-1));
y=dX(2)*(0:(MX(2)-1));
U_1 = U_calcul(:,1,:);
U_1 = reshape(U_1,[MX 2]);
figure;imagesc(x,y,U_1(:,:,1)');axis xy; axis equal; colorbar;

Q = idivide(N_tot,nb_file);
R = mod(N_tot,nb_file);

for i = 1:nb_file
    U= U_calcul(:,(i-1)*Q+1:i*Q,:);
    save( [ '/home/guillaume.lepape@eurogiciel.fr/HDD/REDLUM_CODE/data/folder_file_temp_test_variance_tensor_2_modes__a_cst__decor_by_subsampl_bt_decor_choice_htgen2_threshold_1fct_test_b/dsamp_11_' num2str(i) '_U_temp'],'U','-v7.3');
end

U = U_calcul(:,nb_file*Q+1:N_tot,:);
save( [ '/home/guillaume.lepape@eurogiciel.fr/HDD/REDLUM_CODE/data/folder_file_temp_test_variance_tensor_2_modes__a_cst__decor_by_subsampl_bt_decor_choice_htgen2_threshold_1fct_test_b/dsamp_11_' num2str(nb_file +1) '_U_temp'],'U','-v7.3');
    
end

