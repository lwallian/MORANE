function [B_Cpp,C_Cpp] = read_Cpp_deter(param,refine)

nb_modes = param.nb_modes;
nb_modes = nb_modes - 1;
file_B = ['../../unitTestITHACALUM/deterministicTerms/deterministicTerms_refine' ...
    num2str(refine) '/ITHACAoutput/Matrices/B_0_' ...
    num2str(param.nb_modes-1) '_0_mat.txt'];

fileID = fopen( file_B );
    
B_Cpp = zeros(nb_modes, nb_modes);
C = textscan(fileID, '%f %f %f %f %f %f %f %f %f' );

for i = 1:nb_modes
    for j = 1:nb_modes
       B_Cpp(i,j) = C{j}(i);
    end
end

C_Cpp = zeros(nb_modes, nb_modes, nb_modes);

folder_C = ['../../unitTestITHACALUM/deterministicTerms/deterministicTerms_refine' ...
    num2str(refine) '/ITHACAoutput/Matrices/C_0_' ...
    num2str(param.nb_modes-1) '_0/'];

for k = 1:nb_modes
    file_C = [folder_C 'C' num2str(k-1) '_mat.txt'];
    fileID = fopen( file_C );
    
    C = textscan(fileID, '%f %f %f %f %f %f %f %f %f' );
    
    for i = 1:nb_modes
        for j = 1:nb_modes
            C_Cpp(i,j,k) = C{j}(i);
        end
    end
    
end

end

