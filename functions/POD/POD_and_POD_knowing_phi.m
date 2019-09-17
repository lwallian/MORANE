function [param, bt]=POD_and_POD_knowing_phi(param_ref)
% Compute the spatial modes phi of the POD, the corresponding temporal coefficients bt,
% the temporal mean m_U, the time subsampling and
% the residual velocity U neglected by the Galerkin projection
%

% Instantiation of global config
global correlated_model;
% Set warning to error to be able to try catch it
% This is to be able to modify the precomputed c matrix file if there is
% no information about its derivative inside the file
s = warning('error', 'MATLAB:LOAD:VariableNotFound');

%% Calculation of c
% c is the two times correlation function of the snapshots
if exist([param_ref.folder_data param_ref.type_data '_pre_c.mat'],'file')==2
    param_ref.name_file_pre_c_blurred = [param_ref.folder_data ...
        param_ref.type_data '_pre_c'];
    % Test whether the correlation matrix file has the one corresponding to
    % the derivative
    if correlated_model
        try
            load(param_ref.name_file_pre_c_blurred, 'c', 'dt_c', 'param');
        catch
            warning('Old pre_c file, recalculating the matrices');
            [c, dt_c, param] = estimateCovarianceMatrices(param_ref);
            param.name_file_pre_c_blurred = [param.folder_data param.type_data '_pre_c'];
            save(param.name_file_pre_c_blurred, 'c', 'dt_c', 'param');
            dt_c = dt_c * prod(param.dX) / param.N_tot;
        end
    else
        load(param_ref.name_file_pre_c_blurred,'c','param');
    end
    warning(s); % restore the warning to old state
    param.d = length(param.dX);
    if isfield(param_ref,'N_particules')
        param.N_particules = param_ref.N_particules ;
    end
    if isfield(param_ref,'N_estim')
        param.N_estim = param_ref.N_estim ;
    end
    param.nb_modes = param_ref.nb_modes ;
    param.big_data = param_ref.big_data ;
    param.a_time_dependant = param_ref.a_time_dependant ;
    if param_ref.a_time_dependant
        param.type_filter_a=param_ref.type_filter_a;
    end
    param.decor_by_subsampl = param_ref.decor_by_subsampl ;
    param.coef_correctif_estim = param_ref.coef_correctif_estim ;
    param.eq_proj_div_free = param_ref.eq_proj_div_free ;
    param.save_all_bi = param_ref.save_all_bi ;
    param.name_file_mode = param_ref.name_file_mode ;
    param.folder_results = param_ref.folder_results ;
    param.folder_data = param_ref.folder_data ;
    param.adv_corrected = param_ref.adv_corrected;
    param.save_bi_before_subsampling = param_ref.save_bi_before_subsampling;
    clear param_ref
else
    if correlated_model
        [c, dt_c, param] = estimateCovarianceMatrices(param_ref);
        param.name_file_pre_c_blurred = [param.folder_data param.type_data '_pre_c'];
        save(param.name_file_pre_c_blurred,'c', 'dt_c','param');
        dt_c = dt_c * prod(param.dX) / param.N_tot;
    else
        [c,param]=fct_c_POD(param_ref);
        param.name_file_pre_c_blurred = [param.folder_data param.type_data '_pre_c'];
        save(param.name_file_pre_c_blurred,'c','param');
    end
end

c=c*prod(param.dX)/param.N_tot;
nb_modes=param.nb_modes;

%% Diagonalization of c
if param.save_all_bi
    % if param.save_all_bi = true, the N Chronos will be computed and saved
    % with N = the number of time steps
    % param.save_all_bi = false in general
    [W,S]=eig(c);
    lambda=diag(S);clear S % singular values : energy of each modes
    lambda=lambda(end:-1:1);
    lambda=max(lambda,0);
    W=W(:,end:-1:1);
else
    if param.save_all_bi
        [W,S]=eig(c);
    else
        [W,S]=eigs(c,nb_modes);
    end
    lambda=diag(S);clear S % singular values : energy of each modes
end
trace_c=trace(c); % total energy of the velocity

% Quantity of energy represented by nb_modes Chronos
el=cumsum(lambda(1:nb_modes))/trace_c;
disp('Cumulative rate of energy (in %) of the first modes');
disp(100*el);
disp(['Here, we use only the ' num2str(nb_modes) ' first modes.']);

%% Computation of the Chronos b(t)
bt=sqrt(param.N_tot) * W * sqrt(diag(lambda)); % temporal modes % N x nb_modes
clear W;

% Force the convention: bt(1,:) > 0
% It is then easier to compare results of several simulation
idx = bt(1,:)< 0;
if  any(idx)
    idx=find(idx);
    bt(:,idx) = - bt(:,idx);
end

%% Eventually save all b_i for offline studies
if param.save_all_bi
    save([ param.folder_results 'modes_bi_' param.type_data ...
        '.mat'],'bt','param');
end

% Keep only the the first nb_modes values
bt=bt(:,1:nb_modes);
lambda=lambda(1:nb_modes);
param.lambda=lambda;


if param.save_bi_before_subsampling
    save([ param.folder_results 'modes_bi_before_subsampling_' param.type_data ...
        '_nb_modes_' num2str(nb_modes) '.mat'],'bt','param');
end
% keyboard;

disp('SVD of c in POD done')

%% Computation of phi
tic
MAX_possible_nb_modes = 100;
bool = false;
k = param.nb_modes;
while (~ bool) && ( k <= MAX_possible_nb_modes )
    name_file_mode_temp=[ param.folder_data 'mode_' param.type_data ...
        '_' num2str(k) '_modes.mat'];
    bool = ( exist(name_file_mode_temp,'file')==2 );
    %     bool = bool || bool_local;
    k = k + 1;
end
% toc
if bool
    k = k -1;
    if k > param.nb_modes
        load(name_file_mode_temp,'phi_m_U');
        phi_m_U(:,(param.nb_modes+1):k,:)= [];
        save(param.name_file_mode,'phi_m_U');
        clear phi_m_U
    end
else
    param=fct_phi_POD(param,bt);
end
clear name_file_mode_temp k bool;
toc
disp('Topos computed')

%% Time subsampling

%%
tic
nn=20;
vect_threshold = 10.^(-((1:nn)-1));
vect_threshold(1) = vect_threshold(1)*0.9;
% vect_threshold = 10.^((1:nn)-1);
vect_n_subsampl_decor = zeros(1,nn);
for k=1:nn
    param_temp=param;
    param_temp.decor_by_subsampl.spectrum_threshold = vect_threshold(k);
    vect_n_subsampl_decor(k) ...
        = fct_cut_frequency(bt,lambda,param_temp);
end
figure333=figure(333);
semilogx(vect_threshold, vect_n_subsampl_decor, '--o');
ax=axis;axnew=ax;
axnew(3)=ax(3)-0.1*(ax(4)-ax(3));axnew(4)=ax(4)+0.1*(ax(4)-ax(3));
axis(axnew);
drawnow;
eval( ['print -depsc ' param.folder_results ...
    'variation_of_n_subsampling_' ...
    param.type_data '_' ...
    'modes_n=' num2str(param.nb_modes) '.eps']);
clear nn vect_threshold vect_n_subsampl_decor param_temp
close(figure333)
disp('Variation of time subsampling done')    
toc
%%


tic
% Choice of the subsampling rate for the correlated and non correlated
% models
if param.decor_by_subsampl.bool
    if ~correlated_model
        switch param.decor_by_subsampl.choice_n_subsample
            case 'auto_shanon'
                param.decor_by_subsampl.n_subsampl_decor = fct_cut_frequency(bt,lambda,param);
            case 'lms'
                param.decor_by_subsampl.tau_corr = max(correlationTimeLMS(c, bt, param.dt), 1);
                param.decor_by_subsampl.n_subsampl_decor = max(floor(correlationTimeLMS(c, bt, param.dt)), 1);
            case 'htgen'
                param.decor_by_subsampl.tau_corr = max(htgenCorrelationTime(c, bt, param.dt), 1);
                param.decor_by_subsampl.n_subsampl_decor = max(floor(htgenCorrelationTime(c, bt, param.dt)), 1);
            case 'truncated'
                param.decor_by_subsampl.tau_corr = max(correlationTimeCut(c, bt), 1);
                param.decor_by_subsampl.n_subsampl_decor = max(floor(correlationTimeCut(c, bt)), 1);
            otherwise
                error('Invalid downsampling method.')
        end
    else
        global tau_ss;
        switch param.decor_by_subsampl.choice_n_subsample
            case 'auto_shanon'
                param.decor_by_subsampl.n_subsampl_decor = fct_cut_frequency(bt, lambda, param);
                param.decor_by_subsampl.test_fct = 'b';
                tau_ss = fct_cut_frequency(bt, lambda, param); % undo the theshold inside the function for this case
                param.decor_by_subsampl.test_fct = 'db';
            case 'lms'
                dbt = diff(bt, 1, 1) ./ param.dt;
                param.decor_by_subsampl.tau_corr = max(correlationTimeLMS(dt_c, dbt, param.dt), 1);
                param.decor_by_subsampl.n_subsampl_decor = max(floor(correlationTimeLMS(dt_c, dbt, param.dt)), 1);
                tau_ss = max(correlationTimeLMS(c, bt, param.dt), 1);
            case 'htgen'
                dbt = diff(bt, 1, 1) ./ param.dt;
                param.decor_by_subsampl.tau_corr = max(htgenCorrelationTime(dt_c, dbt, param.dt), 1);
                param.decor_by_subsampl.n_subsampl_decor = max(floor(htgenCorrelationTime(dt_c, dbt, param.dt)), 1);
                tau_ss = max(htgenCorrelationTime(c, bt, param.dt), 1);
            case 'truncated'
                dbt = diff(bt, 1, 1) ./ param.dt;
                param.decor_by_subsampl.tau_corr = max(correlationTimeCut(dt_c, dbt), 1);
                param.decor_by_subsampl.n_subsampl_decor = max(floor(correlationTimeCut(dt_c, dbt)), 1);
                tau_ss = max(correlationTimeCut(c, bt), 1);
            otherwise
                error('Invalid downsampling method.')
        end
    end
end
clear c dt_c;

% Subsampling rate
n_subsampl_decor = param.decor_by_subsampl.n_subsampl_decor

%%  Test if the simulation with the same set of parameter
%%% but with non-stationnary variance tensor has already been done

% param_temp = param;
% param_temp.a_time_dependant=true;
% name_file_temp = fct_file_save_1st_result(param_temp);

param = fct_name_file_noise_cov(param);
param_temp = param;
param_temp.a_time_dependant = true;
param = fct_name_file_diffusion_mode(param);
param_temp = fct_name_file_diffusion_mode(param_temp);

bool = (exist(param.name_file_noise_cov,'file')==2 ) && ( ...
    (exist(param.name_file_diffusion_mode,'file')==2) || ...
    (exist(param_temp.name_file_diffusion_mode,'file')==2) );
% bool = (exist(param.name_file_diffusion_mode,'file')==2) || ...
%         (exist(param_temp.name_file_diffusion_mode,'file')==2);

global computed_PIV_variance_tensor
if computed_PIV_variance_tensor 
    bool =true;
    warning('To remove after testing')
end
global compute_fake_PIV
if compute_fake_PIV 
    bool =true;
    warning('To remove after testing')
end
global compute_PIV_modes
if compute_PIV_modes 
    bool =true;
    warning('To remove after testing')
end

% Subsample residual velocity
if ~ bool
    % if ~ ( exist(name_file_temp,'file')==2 )
    %     if n_subsampl_decor > 1
    % Subsample snapshots
    param = sub_sample_U(param);
else
    param = gen_file_U_temp(param);
end

if  strcmp(param.decor_by_subsampl.meth,'bt_decor')
    % Change the time period
    param.dt=n_subsampl_decor*param.dt;
    % Subsample Chronos
    bt=bt(1:param.decor_by_subsampl.n_subsampl_decor:end,:);
    % Change total numbers of snapshots
    param.N_tot=ceil(param.N_tot/n_subsampl_decor);
    param.N_test=ceil((param.N_test+1)/n_subsampl_decor)-1;
end
toc
disp('Subsampling done')
%% Residual velocity
if ~ bool
    tic
    param = residual_U(param,bt);
    toc
    disp('Residual velocity computed')
end

end