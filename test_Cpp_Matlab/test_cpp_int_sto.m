%% Script test portage C++ : integration temporelle sto
init
rng('default');
path = ['/home/guillaume.lepape@eurogiciel.fr/HDD/Guillaume_Le_Pape_data/unitTestITHACALUM/temporal_integration/'];

dt = 1
% nu = 
nModesU = 2
%nb_realiz_vect = [2 1e1 1e2  ]
% nb_realiz_vect = [2 1e1 1e2 1e3 1e4 ]
% nb_realiz_vect = [2 1e1 1e2 1e3 1e4 1e5 ]
nb_realiz_vect = [2 1e1 1e2 1e3 1e4 1e5 1e6]

for i = 1:nModesU
    %B_1_vector(i,1) = 0;
     B_1_vector(i,1) = cos(i);
end
B_1_vector = - B_1_vector;

for i = 1:nModesU
    for j = 1:nModesU
 %       B_matrix(i,j) = 0;
         B_matrix(i,j) = sin((i-10)*(j-20));
    end
end
B_matrix(1,1) =1;

B_matrix = permute( B_matrix, [ 2 1 ] );
B_matrix = - B_matrix;

for i =1:nModesU
    for j = 1:nModesU
        for k = 1:nModesU
%            C_tensor(i,j,k) = 0;
             C_tensor(i,j,k) = exp(-abs((i-1)*(j-2)*(k-3) ));
        end
    end
end
C_tensor = permute( C_tensor, [ 2 3 1 ] );


for i = 1:(nModesU*(nModesU + 1))
    
    for j = 1:(nModesU*(nModesU + 1))
        
%        N_matrix(i,j) = 0;
         N_matrix(i,j) = ((i-2)^2 + (j)^2)/100;
        
    end
end

for j = 1:nModesU
    bt0base(1,j) = 0;
%     bt0base(1,j) = (j)*cos(1);
end
bt0base(1,1) = 1;

for nb_realiz = nb_realiz_vect
    clear bt_tp1 bt_tp1_mm

    nb_realiz
    bt0=repmat(bt0base,[1 1 nb_realiz]);
    
    
    bt_tp1 = ...
        evol_forward_bt_MCMC(...
        B_1_vector,B_matrix,C_tensor, ...
        N_matrix, dt, bt0 );
    m_bt_tp1 = mean(bt_tp1,3);
    bt_tp1_mm = bt_tp1 ;
%     bt_tp1_mm = bt_tp1 - m_bt_tp1;
    cov_bt_tp1 = mean( bt_tp1_mm .* permute ( bt_tp1_mm , [2 1 3] ) ,3);
%     cov_bt_tp1 = (nb_realiz/(nb_realiz-1)) * cov_bt_tp1 ;
    save(['moment_bt1_ ' num2str(nb_realiz) '.txt'],...
        'm_bt_tp1','cov_bt_tp1','-ascii')
    m_bt_tp1
%     keyboard;
end

%% C++
for nb_realiz = nb_realiz_vect
    nb_realiz
    clear bt_tp1 bt_tp1_mm
    bt0cpp = nan( [ 1, nModesU, nb_realiz ]);
    bt_tp1 = nan( [ 1, nModesU, nb_realiz ]);

    for k = 1:nb_realiz
        filename = [ path 'ITHACAoutput/Reduced_coeff_2_1_' ...
            num2str(nb_realiz) '_neglectedPressure/' ...
            'approx_temporalModes_U_' num2str(k-1) '_mat.txt'];
        
%         % filename = 'C:\Users\valentin.resseguier\Downloads\ITHACAoutput\Reduced_coeff_2_0.005_10_neglectedPressure\approx_temporalModes_U_0_mat.txt';
%         delimiter = ' ';
%         startRow = 1;
%         formatSpec = '%f%f%[^\n\r]';
%         fileID = fopen(filename,'r');
%         dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, ...
%             'MultipleDelimsAsOne', true, 'TextType', 'string', ...
%             'HeaderLines' ,startRow-1, 'ReturnOnError', false, 'EndOfLine', '\r\n');
%         fclose(fileID);
%         bt0cpp(1,1,k) = dataArray{:, 1};
%         bt0cpp(1,2,k) = dataArray{:, 2};


        opts = delimitedTextImportOptions("NumVariables", 2);
        opts.DataLines = [1, 1];
        opts.Delimiter = " ";
        opts.VariableNames = ["VarName1", "VarName2"];
        opts.VariableTypes = ["double", "double"];
        opts.ExtraColumnsRule = "ignore";
        opts.EmptyLineRule = "read";
        opts.ConsecutiveDelimitersRule = "join";
        opts.LeadingDelimitersRule = "ignore";
        approxtemporalModesU0mat = readtable(filename, opts);
        approxtemporalModesU0mat = table2array(approxtemporalModesU0mat);
        bt0cpp(1,1,k) = approxtemporalModesU0mat(1);
        bt0cpp(1,2,k) = approxtemporalModesU0mat(2);
        bt0cpp(1,:,k)-bt0base
        clear opts
        
        % filename = 'C:\Users\valentin.resseguier\Downloads\ITHACAoutput\Reduced_coeff_2_0.005_10_neglectedPressure\approx_temporalModes_U_0_mat.txt';
        delimiter = ' ';
        startRow = 2;
        formatSpec = '%f%f%[^\n\r]';
        fileID = fopen(filename,'r');
        dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, ...
            'MultipleDelimsAsOne', true, 'TextType', 'string', ...
            'HeaderLines' ,startRow-1, 'ReturnOnError', false, 'EndOfLine', '\r\n');
        fclose(fileID);
        bt_tp1(1,1,k) = dataArray{:, 1};
        bt_tp1(1,2,k) = dataArray{:, 2};
        clearvars filename delimiter startRow formatSpec fileID dataArray ans;
        
%         keyboard;
    end
    
    m_bt_tp1_cpp = mean(bt_tp1,3);
    bt_tp1_mm = bt_tp1 ;
%     bt_tp1_mm = bt_tp1 - m_bt_tp1_cpp;
    cov_bt_tp1_cpp = mean( bt_tp1_mm .* permute ( bt_tp1_mm , [2 1 3] ) ,3);
%     cov_bt_tp1_cpp = (nb_realiz/(nb_realiz-1)) * cov_bt_tp1_cpp ;
    save(['moment_bt1_ ' num2str(nb_realiz) '_cpp.txt'],...
        'm_bt_tp1_cpp','cov_bt_tp1_cpp','-ascii')
    m_bt_tp1_cpp
%     keyboard;
end

%% Comparaison

for i_nb_realiz = 1:length(nb_realiz_vect)
    nb_realiz = nb_realiz_vect(i_nb_realiz);
    clear bt_tp1 bt_tp1_mm 
    clear m_bt_tp1_cpp cov_bt_tp1_cpp m_bt_tp1_cpp cov_bt_tp1_cpp
    %%
    filename = ['moment_bt1_ ' ...
        num2str(nb_realiz) '.txt'];
    delimiter = ' ';
    endRow = 1;
    formatSpec = '%f%f%[^\n\r]';
    fileID = fopen(filename,'r');
    dataArray = textscan(fileID, formatSpec, endRow, ...
        'Delimiter', delimiter, 'MultipleDelimsAsOne', true, ...
        'TextType', 'string', 'ReturnOnError', false, 'EndOfLine', '\r\n');
    fclose(fileID);
    m_bt_tp1 = [dataArray{1:end-1}];
    clearvars endRow dataArray ans;
    
    startRow = 2;
    fileID = fopen(filename,'r');
    dataArray = textscan(fileID, formatSpec, ...
        'Delimiter', delimiter, 'MultipleDelimsAsOne', true, ...
        'TextType', 'string', 'HeaderLines' ,startRow-1, ...
        'ReturnOnError', false, 'EndOfLine', '\r\n');
    fclose(fileID);
    cov_bt_tp1 = [dataArray{1:end-1}];
    clearvars filename startRow fileID dataArray ans;
    
    filename = ['moment_bt1_ ' ...
        num2str(nb_realiz) '_cpp.txt'];
    delimiter = ' ';
    endRow = 1;
    formatSpec = '%f%f%[^\n\r]';
    fileID = fopen(filename,'r');
    dataArray = textscan(fileID, formatSpec, endRow, ...
        'Delimiter', delimiter, 'MultipleDelimsAsOne', true, ...
        'TextType', 'string', 'ReturnOnError', false, 'EndOfLine', '\r\n');
    fclose(fileID);
    m_bt_tp1_cpp = [dataArray{1:end-1}];
    clearvars endRow dataArray ans;
    
    startRow = 2;
    fileID = fopen(filename,'r');
    dataArray = textscan(fileID, formatSpec, ...
        'Delimiter', delimiter, 'MultipleDelimsAsOne', true, ...
        'TextType', 'string', 'HeaderLines' ,startRow-1, ...
        'ReturnOnError', false, 'EndOfLine', '\r\n');
    fclose(fileID);
    cov_bt_tp1_cpp = [dataArray{1:end-1}];
    clearvars filename startRow fileID dataArray ans;
%%
    m_bt_tp1
    m_bt_tp1_cpp
    %m_bt_tp1-bt0base
    %m_bt_tp1_cpp-bt0base
    
    err_m_bt_tp1 = m_bt_tp1 - m_bt_tp1_cpp ;
    err_cov_bt_tp1 = cov_bt_tp1 - cov_bt_tp1_cpp ;
    norm_err_m_bt_tp1(i_nb_realiz) = sqrt( sum( err_m_bt_tp1(:).^2)) 
    norm_err_cov_bt_tp1(i_nb_realiz) = sqrt( sum( err_cov_bt_tp1(:).^2)) ;
%     keyboard;
end

figure(1);
loglog(nb_realiz_vect,norm_err_m_bt_tp1);
hold on
loglog(nb_realiz_vect,nb_realiz_vect.^(-1/2));
hold off
xlabel('nb realizations')
ylabel('Chronos mean distance');

figure(2);
loglog(nb_realiz_vect,norm_err_cov_bt_tp1);
hold on
loglog(nb_realiz_vect,nb_realiz_vect.^(-1/2));
hold off
xlabel('nb realizations')
ylabel('Chronos covariance distance');

m_bt_tp1
m_bt_tp1_cpp



