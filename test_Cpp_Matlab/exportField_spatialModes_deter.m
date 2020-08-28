function U = exportField_spatialModes_deter(param)
MX = param.MX;
M = param.M;
nb_modes = param.nb_modes;
d = param.d;
dX = param.dX;

folder = param.name_file_folder;

U = zeros(M,nb_modes+1,d);

for k = 1:nb_modes
        
    fileID = fopen( [folder num2str(k-1)],'r' );
    
    C = textscan(fileID,'%f %f %f %f %f');
  
    for i = 1:MX(1)
        for j = 1:MX(2)
            for l = 1:d
                x = C{1}(i + MX(1)*(j-1) + M*(l-1));
                y = C{2}(i + MX(1)*(j-1) + M*(l-1));
                z = C{3}(i + MX(1)*(j-1) + M*(l-1));
                Ux = C{4}(i + MX(1)*(j-1) + M*(l-1));
                Uy = C{5}(i + MX(1)*(j-1) + M*(l-1));
       
                if ( abs(z-1) >= 1e-2)
                    floor_x = abs( floor(x/dX(1)) - (x/dX(1)) );
                    floor_y = abs( floor(y/dX(2)) - (y/dX(2)) );
                    ceil_x = abs( ceil(x/dX(1)) - (x/dX(1)) );
                    ceil_y = abs( ceil(y/dX(2)) - (y/dX(2)) );
        
                    % fprintf('floor_x : %0.8f \n',floor_x);
                    % fprintf('floor_y : %0.8f \n',floor_y);
                    % fprintf('ceil_x : %0.8f \n',ceil_x);
                    % fprintf('ceil_y : %0.8f \n',ceil_y);

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
                    
                    U(ind_x + MX(1)*(ind_y-1),k,:) = [Ux Uy];
                end
            end
        end
    end
    fclose(fileID);
end

end

