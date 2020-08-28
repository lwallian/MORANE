function [param] = exportFields_globalTest(param, nb_file)

folder = [pwd '/../../unitTestITHACALUM/globalTest/pointField'];

%threshold_str = num2str(param.decor_by_subsampl.spectrum_threshold);
%iii = (threshold_str=='.');
%threshold_str(iii)='_';

MX = param.MX;
dX = param.dX;
M = param.M;
N_tot = param.N_tot;
d = param.d;
Re = 100;
dt = 0.05;
normalized = true;

folder_data = [ pwd '/../data/folder_DNS100_OpenFOAM_2D_2020' ];

mkdir(folder_data);

file = 60220;



Q = idivide(N_tot,nb_file);
R = mod(N_tot,nb_file);
Q 
R
%{
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

    fileID = fopen( [folder '/U_' file_format],'r' );

    [folder '/U_' file_format]
    
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
    
    file = file + 5;
end
    
x=dX(1)*(0:(MX(1)-1));
y=dX(2)*(0:(MX(2)-1));
U_1 = U_calcul(:,1,:);
U_1 = reshape(U_1,[MX 2]);
figure;imagesc(x,y,U_1(:,:,1)');axis xy; axis equal; colorbar;

%}



for l = 1:nb_file
    
    U = zeros(M,Q,d);
    
    for k = 1:Q
        Ndigits = dec2base(file,10) - '0';
    
        if (Ndigits(end) ~= 0)
            file_format = sprintf('%0.2f',file*10^-2);
        elseif (Ndigits(end-1) == 0)
            file_format = sprintf('%0.0f',file*10^-2);
        else
            file_format = sprintf('%0.1f',file*10^-2);
        end
    
        file_format

        fileID = fopen( [folder '/U_' file_format],'r' );
    
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
    
                U(ind_x + MX(1)*(ind_y-1),k,:) = [Ux Uy];
            end
        end
        fclose(fileID);
    
        file = file + 5;
    end
    
    
    x=dX(1)*(0:(MX(1)-1));
    y=dX(2)*(0:(MX(2)-1));
    U_1 = U(:,1,:);
    U_1 = reshape(U_1,[MX 2]);
    figure;imagesc(x,y,U_1(:,:,1)');axis xy; axis equal; colorbar;
    
    U = reshape(U ,[MX Q d]);
    
    save( [ folder_data '/file_DNS100_OpenFOAM_2D_2020_run_' num2str(l) ],'Re','U','dX','dt','normalized','-v7.3');
    
    
end



    
end

