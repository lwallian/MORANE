function [L_Cpp, S_Cpp] = read_Cpp_sto(param,refine)

nb_modes = param.nb_modes;
nb_modes = nb_modes - 1;
file_L = ['../../unitTestITHACALUM/stoTerms/stoTerms_refine' num2str(refine) '/ITHACAoutput/Matrices/L_' num2str(nb_modes) '_mat.txt'];

fileID = fopen( file_L );
    
L_Cpp = zeros(nb_modes, nb_modes);
C = textscan(fileID, '%f %f %f %f %f %f %f %f %f' );

for i = 1:nb_modes
    for j = 1:nb_modes
       L_Cpp(i,j) = C{j}(i);
    end
end

file_S = ['../../unitTestITHACALUM/stoTerms/stoTerms_refine' num2str(refine) '/ITHACAoutput/Matrices/S_' num2str(nb_modes) '_mat.txt'];

fileID = fopen( file_S );
    
S_Cpp = zeros(nb_modes, nb_modes);
C = textscan(fileID, '%f %f %f %f %f %f %f %f %f' );

for i = 1:nb_modes
    for j = 1:nb_modes
       S_Cpp(i,j) = C{j}(i);
    end
end



end

