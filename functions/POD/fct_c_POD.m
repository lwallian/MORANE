function [c, param]=fct_c_POD(param_ref,bool_init)
% Compute c the matrix (two times correlation function) defined by 
% the snapshots method (Sirovich)
%

global correlated_model;

if param_ref.data_in_blocks.bool % if data are saved in several files
    % Get some information on how the data are saved
    param_blocks = read_data_blocks(param_ref.type_data, param_ref.folder_data);
    % number of files used to save data
    nb_blocks = param_blocks.data_in_blocks.nb_blocks;
    % number of snapshots by file
    len_blocks = param_blocks.data_in_blocks.len_blocks;
    % total number of snapshots
    N_tot = len_blocks * nb_blocks;
    param_ref.name_file_U_centered = {};
%     param_ref.name_file_U_temp={};
    
    % Temporary parameters
    param_ref_temp = param_ref;
    param_ref_temp.data_in_blocks.type_whole_data = param_ref.type_data;
    param_ref_temp.data_in_blocks.bool = false;
    
    % Initialization of c
    c = nan(N_tot);
    % TO DO : parallelize thes two loops 
    for big_Ti = 1 : nb_blocks % loop on files
        % To know if you are at the beginning of the loop
        bool_init_temp(1) = (big_Ti == 1); 
        for big_Tj = 1 : nb_blocks % loop on files
            big_Ti
            big_Tj
            tic;
            % To know if you are at the beginning of the loop
            bool_init_temp(2) = (big_Tj == 1);
            % Set which file will be used for the lines of the block
            param_ref_temp.type_data = ...
                [param_ref.type_data num2str(big_Ti)];
            % Set which file will be used for the columns of the block
            param_ref_temp.data_in_blocks.type_data2 = ...
                [param_ref.type_data num2str(big_Tj)];
            % Set lines indexes of the block
            vi=(big_Ti-1)*len_blocks +1 : (big_Ti)*len_blocks;
            % Set columns indexes of the block
            vj=(big_Tj-1)*len_blocks +1 : (big_Tj)*len_blocks;
            if big_Tj >= big_Ti
                % Compute and assign values in the block
                [c(vi, vj), param_temp] = fct_c_POD(param_ref_temp, bool_init_temp);
            else
                c(vi, vj) = c(vj, vi)';
            end
            
            % Replace param by param_temp (computed above) after the first
            % block calculation
            if all(bool_init_temp)
                param = param_temp;
            end
            toc;
        end
        % Create (iteratively) a cell containing the names of files, 
        % which contain residual velocities 
        param_ref.name_file_U_centered = [param_ref.name_file_U_centered ...
            {param_temp.name_file_U_centered}];
%         param_ref.name_file_U_temp=[param_ref.name_file_U_temp ...
%             {param_temp.name_file_U_temp}];
    end
    
    % Assign output parameters values
    
    % Cell containing the names of files, which contain residual velocities 
    param.name_file_U_centered = param_ref.name_file_U_centered;
%     param.name_file_U_temp=param_ref.name_file_U_temp;
    param.data_in_blocks.bool = true;
    param.data_in_blocks.nb_blocks = nb_blocks;
    param.data_in_blocks.len_blocks = len_blocks;
    param.type_data = param_ref.type_data;
    param.data_in_blocks.type_data2 = nan;
    param.N_tot = N_tot; % total number of snapashots 
    param.N_test = N_tot - 1; % total number of snapashots to reconstruct
    
    disp('Estimation of c in POD completed')
else
    % Here, only 1 or 2 files are considered
    % Computation of the autocorrelation of the single file or the cross
    % correlation of the two files
    % The two files will be called left file U1 and right file U
    
    % bool_init(1) = true if it is the left file is the first file
    % bool_init(2) = true if it is the right file is the first file
    if nargin < 2
       bool_init = true(1, 2) ; 
    end
    
    % Path for the file which contains or will contain residual velocity of
    % the left file
    name_file_U_centered = [param_ref.folder_data param_ref.type_data '_U_centered'];
%     name_file_U_temp=[param_ref.folder_data param_ref.type_data '_U_temp'];
    
%     if ~all(bool_init) 
    % TO DO : enable using pre-computed time average
    if ~all(bool_init) ...
            || (exist([name_file_U_centered, '.mat'], 'file') == 2)
        % load left file U1
        load(name_file_U_centered, 'U');
%         load(name_file_U_temp);
        U1 = U;
        clear U;
        if all(bool_init)
            % Load the âram file
            if isfield(param_ref.data_in_blocks, 'type_whole_data') % if data are saved in several files
                param = read_param_data(param_ref.type_data, param_ref.folder_data, ...
                    param_ref.data_in_blocks.type_whole_data);
            else
                param = read_param_data(param_ref.type_data, param_ref.folder_data);
            end
            param_ref = rmfield(param_ref, 'type_data');
            param_ref = rmfield(param_ref, 'folder_data');
            param = mergestruct(param, param_ref);
        else
            param = param_ref;
        end
        param.name_file_U_centered = name_file_U_centered;
%         param.name_file_U_temp=name_file_U_temp;
    else % First iteration: the time average value needs to be computed first
        % Compute time average value on all files and save it
        % TO DO : enable using precalculated average
        fct_m_U(param_ref);
        % Load data of the left file U1 and parameters
        % and remove time average value from U1
        % TO DO : enable using pre-treated snapshots
        [U1,param] = pre_treatement_of_U(param_ref);
    end
    big_data = param.big_data;
    % M is the number of points in the spatial grid
    % N is the number of snapshots on the left file U1
    % d = 2 or 3 is the dimension of the space
    [param.M , param.N_tot, param.d] = size(U1);
    d = param.d;
    
    % If working with the correlated model
    if correlated_model == true && ...
            ~strcmp(param.decor_by_subsampl.choice_n_subsample, 'auto_shanon')
        U1 = diff(U1, 2);
        param.N_tot = param.N_tot - 1; % REVISER
    end 
    
    if strcmp(param.type_data, param.data_in_blocks.type_data2)
        % if left file = right file
        % An autocorrelation is computed
        U = U1;
    else
        % if left file ~= right file
        % An crosscorrelation is computed
        
        % Name of the right file
        name_file_U_centered2 = [param.folder_data param.data_in_blocks.type_data2, '_U_centered'];
%         name_file_U_temp2=[param.folder_data param.data_in_blocks.type_data2 '_U_temp'];
%         if ~bool_init(1) % all files have been pre-treated
        if ~bool_init(1) ...  % all files have been pre-treated
                || (exist([name_file_U_centered2, '.mat'], 'file') == 2)
            load(name_file_U_centered2);
%             load(name_file_U_temp2);
        else % the right file has not already been pre-treated
            param2 = param_ref;
            param2.type_data = param.data_in_blocks.type_data2;
            % Load data of the left file U1 and parameters
            % and remove time average value from U1
            % TO DO : enable using pre-treated snapshots
            U = pre_treatement_of_U(param2);
            clear param2
        end
    end
    
    %% Calculation of c
    
    if param.N_tot > param.M
        warning('The computing of the mode phi and the coefficients bt should use the fact that N > M');
    end
    
    % Initialization
    c = zeros(param.N_tot);
    
    % Calculation of the correlation matrix
    if big_data
        for i = 1 : param.N_tot % loop on time
            for k = 1: d % loop on the dimension
                U1_temp = U1(:, i, k);
                for j = 1 : param.N_tot
                    c(i, j) = c(i, j) + U1_temp' * U(:, j, k);
                end
%                 parfor j=1:param.N_tot
%                     c(i,j)=c(i,j)+U1_temp'*U(:,j,k);
%                 end
            end
        end
    else
        for k = 1 : d
            c = c + U1(:, :, k)' * U(:, :, k);
        end
    end    
    
end