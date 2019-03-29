function param = fct_def_param(nb_modes,igrida,coef_correctif_estim,save_all_bi,decor_by_subsampl,a_time_dependant)


%clear all;
clear param;
igrida = false;
%     param.type_data='incompact3D_noisy2D_40dt_subsampl';
%param.type_data='incompact3D_noisy2D';
%     param.type_data = 'incompact3d_wake_episode3_cut';
param.type_data = 'adda';

if igrida
    %param.folder_data = '/temp_dd/igrida-fs1/sokloeun/data/';
    %param.folder_results = '/temp_dd/igrida-fs1/sokloeun/results/';
    param.folder_data = '/temp_dd/igrida-fs1/sohuot/blurred_data/';
    param.folder_results = '/temp_dd/igrida-fs1/sohuot/results/';
    param.folder_pre_U_temp = '/temp_dd/igrida-fs1/sohuot/pre_U_temp/';
    param.folder_pre_c_blurred ='/temp_dd/igrida-fs1/sohuot/pre_c_blurred/'
    param.big_data =true;
    plot_bts = false;
else
    param.folder_U_t_test = [ pwd '/U_t_test/'];
    param.folder_pre_U_temp = [ pwd '/pre_U_temp/'];
    param.folder_data ='/Users/vressegu/Documents/matlab/POD-NS_Stochastique/current_used/data/';
    %     param.folder_U_t_test ='/Users/sokloeun/Documents/matlab/POD_full_sto_model/U_t_test/';
    %     param.folder_pre_U_temp ='/Users/sokloeun/Documents/matlab/POD_full_sto_model/pre_U_temp/';
    %     param.folder_data = '/Users/sokloeun/Documents/matlab/POD_full_sto_model/data/';
    param.folder_results =  [ pwd '/resultats/current_results/'];
end


param.nb_modes = nb_modes;

param.big_data = false;

% Model for the tensor a
if ~exist('a_time_dependant','var')
    param.a_time_dependant=false;
else % the variable is already defined in a super_main
    param.a_time_dependant=a_time_dependant;
end


%% Parameters already chosen
% Do not modify the following lines

% % Model for the tensor a
% if param.a_time_dependant
%     param.type_filter_a='b_i';
% end

% Parameters used if data are saved in different files
switch param.type_data
    case {'inc3D_Re300_40dt_blocks','inc3D_Re3900_blocks'}
        % data_in_blocks.bool = trues if data are saved in different files
        param.data_in_blocks.bool = true;
        param.big_data=true;
    otherwise
        param.data_in_blocks.bool = false;
        % param.data_in_blocks.nb_blocks is the number of files used to
        % save the data
        param.data_in_blocks.nb_blocks=1;
        param.data_in_blocks.type_data2=param.type_data;
end

% Parameters used for the time subsampling of data
if ~exist('decor_by_subsampl','var')
    param.decor_by_subsampl.bool=true;
    %         param.decor_by_subsampl.bool=false;
    if param.decor_by_subsampl.bool
        % param.decor_by_subsampl.n_subsampl_decor is the new time sampling
        % period dived by the former one.
        % param.decor_by_subsampl.n_subsampl_decor = nan if you let the
        % algorithm choose this value
        param.decor_by_subsampl.n_subsampl_decor=nan;
        %         param.decor_by_subsampl.n_subsampl_decor=10;
        % param.decor_by_subsampl.meth determines at which step of the
        % algortihm, there should be a time subsampling
        % POD_decor : at the beginning
        % bt_decor : after estimating the Chronos bt
        % a_estim_decor : after the variance tensor a
        param.decor_by_subsampl.meth='bt_decor'; % 'POD_decor' 'bt_decor' 'a_estim_decor'
        % param.decor_by_subsampl.choice_n_subsample determines the way of
        % choosing the new time sampling period
        % 'auto_shanon' means that it use a Nyquist-Shanon based criterion
        % 'tuning' means that param.decor_by_subsampl.n_subsampl_decor is
        % set manually
        param.decor_by_subsampl.choice_n_subsample='auto_shanon';
        % param.decor_by_subsampl.spectrum_threshold and param.decor_by_subsampl.test_fct
        % are parameters used in the new time sampling period choice.
        % spectrum_threshold is a threshold to know
        % when a Chronos spectrum is equal to zero
        param.decor_by_subsampl.spectrum_threshold=1e-4;
        %         param.decor_by_subsampl.spectrum_threshold=1/1000;
        % test_fct determines if the Shanon criterion is used on the derivatives
        % of Chronos ('db') or on quadratic functions of Chronos ('b')
        param.decor_by_subsampl.test_fct = 'b';%'b';'db';
    end
else % the variable is already defined in a super_main
    param.decor_by_subsampl=decor_by_subsampl;
end

% If coef_correctif_estim.learn_coef_a = true, corrective coefficients are
% estimated to modify the values of the variance tensor a, in order to
% improve the results
param.coef_correctif_estim.learn_coef_a=false;

% param.eq_proj_div_free=true if the PDE is
% projected on the free divergence space
param.eq_proj_div_free=true;

% if param.save_all_bi = true, the N Chronos will be computed and saved
% with N = the number of time steps
% It can be useful for offline studies
if ~exist('save_all_bi','var')
    param.save_all_bi=false;
else % the variable is already defined in a super_main
    param.save_all_bi=save_all_bi;
    clear save_all_bi
end

% If plot_bts = true the Chronos will be plotted
plot_bts = true;

% Set directories to load and save data
if igrida
    param.folder_data = '/temp_dd/igrida-fs1/vressegu/data/';
    param.folder_results = '/temp_dd/igrida-fs1/vressegu/results/';
    param.big_data =true;
    plot_bts = false;
else
    param.folder_data = '/Users/vressegu/Documents/matlab/POD-NS_Stochastique/current_used/data/';
    param.folder_results =  [ pwd '/resultats/current_results/'];
end
param.name_file_mode=['mode_' param.type_data '_' num2str(param.nb_modes) '_modes.mat'];
param.igrida=igrida; clear igrida
param.name_file_mode=[ param.folder_data param.name_file_mode ];
