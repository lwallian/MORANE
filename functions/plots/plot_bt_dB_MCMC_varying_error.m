function [ idx_min_error, idx_min_err_tot] = ...
    plot_bt_dB_MCMC_varying_error(...
    param,bt_forecast_sto_scalar,bt_forecast_sto_beta,...
    bt_pseudoSto, bt_forecast_sto_a_NC_modal_dt, bt_forecast_deter, ...
    bt_forecast_MEV,bt_forecast_MEV_noise,struct_bt_MEV_noise,bt_tot,struct_bt_MCMC,bt_MCMC)
% Plot the sum of the error along time (in log scale)
%

clear height
% logscale =true
logscale =false

LineWidth = 1;
% FontSize = 6;
% FontSizeTtitle = 7;
% width=0.7;
% height=0.45;
FontSize = 10;
FontSizeTtitle = 11;
width=1;
height=0.7;

height = height*3/2;

% width=2;
% height=1.5;
% % width=6;
% % height=2.4;
% % % width=2;
% % % height=1.6;
% % % % width=2.5;
% % % % height=1.6;
% % % % % width=1.5;
% % % % % height=1.2;

% width=2;
% % width=1.5;
% height=1.5;
% % height=2.2;
X0=[0 0];
% 2 et 1.8

plot_deter=param.plot.plot_deter;
plot_EV=param.plot.plot_EV;
plot_tuned=param.plot.plot_tuned;
plot_modal_dt=param.plot_modal_dt;

param.param.nb_modes = size(bt_tot,2);
param.type_data
dt_tot=param.dt;

if isfield(param,'N_tot')
    N_tot=param.N_tot;
    N_test=param.N_test;
else
    N_tot=300;
    N_test=299;
end

if nargin < 12
    bt_MCMC =nan(size(bt_tot));
end
if param.plot_EV_noise
    bt_forecast_MEV_noise=bt_forecast_MEV_noise(1:N_test,:,:);
    struct_bt_MEV_noise.tot.mean=struct_bt_MEV_noise.tot.mean(1:N_test,:);
    struct_bt_MEV_noise.tot.var=struct_bt_MEV_noise.tot.var(1:N_test,:);
    struct_bt_MEV_noise.tot.one_realiz=struct_bt_MEV_noise.tot.one_realiz(1:N_test,:);
end
bt_MCMC=bt_MCMC(1:N_test,:,:);
struct_bt_MCMC.tot.mean=struct_bt_MCMC.tot.mean(1:N_test,:);
struct_bt_MCMC.tot.var=struct_bt_MCMC.tot.var(1:N_test,:);
struct_bt_MCMC.tot.one_realiz=struct_bt_MCMC.tot.one_realiz(1:N_test,:);
% struct_bt_MCMC.fv.mean=struct_bt_MCMC.fv.mean(1:N_test,:);
% struct_bt_MCMC.fv.var=struct_bt_MCMC.fv.var(1:N_test,:);
% struct_bt_MCMC.fv.one_realiz=struct_bt_MCMC.fv.one_realiz(1:N_test,:);
% struct_bt_MCMC.m.mean=struct_bt_MCMC.m.mean(1:N_test,:);
% struct_bt_MCMC.m.var=struct_bt_MCMC.m.var(1:N_test,:);
% struct_bt_MCMC.m.one_realiz=struct_bt_MCMC.m.one_realiz(1:N_test,:);

% BETA : confidence interval
% struct_bt_MCMC.qtl = struct_bt_MCMC.qtl(1:N_test,:);
% struct_bt_MCMC.diff = struct_bt_MCMC.diff(1:N_test,:);
% end BETA

% struct_bt_MCMC=struct_bt_MCMC(1:N_test,:);
bt_tot=bt_tot(1:N_test,:);
bt_forecast_deter=bt_forecast_deter(1:N_test,:);
bt_forecast_MEV=bt_forecast_MEV(1:N_test,:);
bt_pseudoSto=bt_pseudoSto(1:N_test,:);
if ~ param.reconstruction
    param.truncated_error2=param.truncated_error2(1:N_test,:);
end
N_test=N_test-1;

dt_tot=param.dt;
N_time_final=N_tot;
time=(1:(N_test+1))*dt_tot;
time_ref = time;

% N_time_final=param.N_tot;
% N_test=param.N_test;
% param.nb_modes=param.param.nb_modes;
% time=(0:(param.N_tot-1))*dt_tot;
% % time=(0:(N_test))*dt_tot;
% % time=(max(1,N_time_final-2*N_test+1):N_time_final)*dt_tot;

%%
if strcmp(param.type_data, 'LES_3D_tot_sub_sample_blurred')
    nrj_varying_tot=18.2705;
    load resultats/nrj_mean_LES3900;
elseif strcmp(param.type_data, 'inc3D_Re300_40dt_blocks_truncated') ...
        || strcmp(param.type_data, 'inc3D_Re300_40dt_blocks')
    param.file_nrj_mU=[ param.folder_results '_nrj_mU_' ...
        param.type_data '0.mat'];
    load(param.file_nrj_mU, 'nrj_mU');
    norm0=nrj_mU;
    param.name_file_pre_c = [param.folder_data param.type_data '_pre_c'];
    param_ref=param;
    load(param.name_file_pre_c,'c','param');
    param_c = param; param=param_ref;clear param_ref;
    %     c=c*prod(param.dX)/(param.N_tot*param.decor_by_subsampl.n_subsampl_decor-1);
    c=c*prod(param.dX)/param_c.N_tot;
    %     load(param.name_file_pre_c,'c');
    %     c=c*prod(param.dX)/(param.N_tot*param.decor_by_subsampl.n_subsampl_decor-1);
    nrj_varying_tot=trace(c);
    
elseif strcmp(param.type_data, 'inc3D_Re3900_blocks') ...
        || strcmp(param.type_data, 'inc3D_Re3900_blocks119')
    %     warning(' the value of nrj_varying_tot is wrong');
    %     nrj_varying_tot=21;
    %     norm0=16;
    load resultats/nrj_mean_DNS3900blurred;
    param.name_file_pre_c = [param.folder_data param.type_data '_pre_c'];
    param_ref=param;
    load(param.name_file_pre_c,'c','param');
    param_c = param; param=param_ref;clear param_ref;
    %     c=c*prod(param.dX)/(param.N_tot*param.decor_by_subsampl.n_subsampl_decor-1);
    c=c*prod(param.dX)/param_c.N_tot;
    %     load([ param.folder_data 'inc3D_Re3900_blocks_pre_c.mat'],'c');
    %     %     c=c*prod(param.dX)/(param.N_tot_old*param.decor_by_subsampl.n_subsampl_decor-1);
    %     c=c*prod(param.dX)/(param.N_tot*param.decor_by_subsampl.n_subsampl_decor-1);
    nrj_varying_tot=trace(c);
    %     keyboard;
elseif strcmp(param.type_data, 'incompact3d_wake_episode3_cut')
    param.file_nrj_mU=[ param.folder_results '_nrj_mU_' param.type_data '.mat'];
    load(param.file_nrj_mU, 'nrj_mU');
    norm0=nrj_mU;
    param.name_file_pre_c = [param.folder_data param.type_data '_pre_c'];
    param_ref=param;
    load(param.name_file_pre_c,'c','param');
    param_c = param; param=param_ref;clear param_ref;
    %     c=c*prod(param.dX)/(param.N_tot*param.decor_by_subsampl.n_subsampl_decor-1);
    c=c*prod(param.dX)/param_c.N_tot;
    %     load(param.name_file_pre_c,'c');
    %     %     load(['resultats/nrj_mean_' param.type_data ' .mat']);
    %     %     load([ param.folder_data param.type_data ' .mat'],'c');
    %     c=c*prod(param.dX)/(param.N_tot*param.decor_by_subsampl.n_subsampl_decor-1);
    nrj_varying_tot=trace(c);
    %     keyboard;
    % elseif strcmp(param.type_data, 'inc3D_Re3900_blocks_truncated')
    %     param.file_nrj_mU=[ param.folder_results '_nrj_mU_' param.type_data '90.mat'];
    %     load(param.file_nrj_mU, 'nrj_mU');
    %     norm0=nrj_mU;
    %     param.name_file_pre_c = [param.folder_data param.type_data '_pre_c'];
    %     load(param.name_file_pre_c,'c');
    % %     load(['resultats/nrj_mean_' param.type_data ' .mat']);
    % %     load([ param.folder_data param.type_data ' .mat'],'c');
    %     c=c*prod(param.dX)/(param.N_tot*param.decor_by_subsampl.n_subsampl_decor-1);
    %     nrj_varying_tot=trace(c);
    % %     keyboard;
    % else
    %     warning('no values for the total energy and the energy of the mean')
    %     norm0=0;
    %     nrj_varying_tot=0;
else
    % BETA if param.data_in_blocks.bool % if data are saved in several files
    % Get some information on how the data are saved
    %         param_blocks=read_data_blocks(param.type_data,param.folder_data);
    %         nb_blocks = param_blocks.data_in_blocks.nb_blocks;
    %         nb_blocks = param.data_in_blocks.nb_blocks;
    %         param.file_nrj_mU=[ param.folder_results '_nrj_mU_' ...
    %             param.type_data num2str(nb_blocks) '.mat'];
    %     else
    param.file_nrj_mU=[ param.folder_results '_nrj_mU_' ...
        param.type_data '0.mat'];
    % BETA end
    if exist(param.file_nrj_mU,'file')==2
        load(param.file_nrj_mU, 'nrj_mU');
    else
        %         if isfield(param.data_in_blocks,'type_whole_data') % if data are saved in several files
        %             type_data_mU=[param.data_in_blocks.type_whole_data num2str(0)];
        %         else
        %             type_data_mU=[param.type_data num2str(0)];
        %         end
        type_data_mU=[param.type_data num2str(0)];
        param.name_file_mU=[param.folder_data type_data_mU '_U_centered'];
        load(param.name_file_mU,'m_U');
        
        switch param.type_data
            case {'incompact3d_wake_episode3_cut_truncated',...
                    'incompact3D_noisy2D_40dt_subsampl',...
                    'incompact3D_noisy2D_40dt_subsampl_truncated',...
                    'inc3D_Re300_40dt_blocks', 'inc3D_Re300_40dt_blocks_truncated', 'inc3D_Re300_40dt_blocks_test_basis',...
                    'inc3D_Re3900_blocks', 'inc3D_Re3900_blocks_truncated', 'inc3D_Re3900_blocks_test_basis',...
                    'DNS300_inc3d_3D_2017_04_02_blocks', 'DNS300_inc3d_3D_2017_04_02_blocks_truncated',...
                    'DNS300_inc3d_3D_2017_04_02_blocks_test_basis',...
                    'test2D_blocks', 'test2D_blocks_truncated', 'test2D_blocks_test_basis',...
                    'small_test_in_blocks', 'small_test_in_blocks_truncated',...
                    'small_test_in_blocks_test_basis',...
                    'DNS100_inc3d_2D_2018_11_16_blocks',...
                    'DNS100_inc3d_2D_2018_11_16_blocks_truncated',...
                    'DNS100_inc3d_2D_2018_11_16_blocks_test_basis',...
                    'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks',...
                    'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_truncated',...
                    'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_test_basis',...
                    'inc3D_HRLESlong_Re3900_blocks',...
                    'inc3D_HRLESlong_Re3900_blocks_truncated',...
                    'inc3D_HRLESlong_Re3900_blocks_test_basis'}
                m_U(:,:,1)=m_U(:,:,1)-1;
        end
        nrj_mU = sum(m_U(:).^2)*prod(param.dX);
        param.file_nrj_mU=[ param.folder_results '_nrj_mU_' param.type_data '.mat'];
        % param.file_nrj_mU=[ param.folder_results '_nrj_mU_' param.type_data '.mat'];
        mkdir(param.folder_results)
        save(param.file_nrj_mU, 'nrj_mU');
    end
    norm0=nrj_mU;
    param.name_file_pre_c = [param.folder_data param.type_data '_pre_c'];
    param_ref=param;
    load(param.name_file_pre_c,'c','param');
    param_c = param; param=param_ref;clear param_ref;
    %     c=c*prod(param.dX)/(param.N_tot*param.decor_by_subsampl.n_subsampl_decor-1);
    c=c*prod(param.dX)/param_c.N_tot;
    nrj_varying_tot=trace(c);
end

nrj_mean = norm0; clear norm0;
nrj_tot=nrj_mean+nrj_varying_tot;

if param.reconstruction
    err_fix =  ((nrj_varying_tot - sum(param.lambda))/nrj_tot) * ...
        ones(param.N_test,1) ;
else
    err_fix =  param.truncated_error2/nrj_tot ;
end
% err_fix =  zeros(size(err_fix));

bt_0 = sum((bt_tot).^2,2)/nrj_tot+err_fix;
% %     keyboard;
% % err_fix =  1- (nrj_mean+sum(param.lambda))/nrj_tot;
% bt_forecast_sto_a_cst_modal_dt = sum((bt_forecast_sto_a_cst_modal_dt-bt_tot).^2,2)/nrj_tot+err_fix;
% bt_forecast_sto_a_NC_modal_dt = sum((bt_forecast_sto_a_NC_modal_dt-bt_tot).^2,2)/nrj_tot+err_fix;
bt_forecast_deter = sum((bt_forecast_deter-bt_tot).^2,2)/nrj_tot+err_fix;
% bt_forecast_sto_scalar = sum((bt_forecast_sto_scalar-bt_tot).^2,2)/nrj_tot+err_fix;
% bt_forecast_sto_beta = sum((bt_forecast_sto_beta-bt_tot).^2,2)/nrj_tot+err_fix;
bt_pseudoSto = sum((bt_pseudoSto-bt_tot).^2,2)/nrj_tot+err_fix;
% bt_sans_coef_a_NC = sum((bt_sans_coef_a_NC-bt_tot).^2,2)/nrj_tot+err_fix;
bt_forecast_MEV = sum((bt_forecast_MEV-bt_tot).^2,2)/nrj_tot+err_fix;
struct_bt_MCMC.tot.mean = sum((struct_bt_MCMC.tot.mean-bt_tot).^2,2)/nrj_tot+err_fix;
struct_bt_MCMC.tot.one_realiz = sum((struct_bt_MCMC.tot.one_realiz-bt_tot).^2,2)/nrj_tot+err_fix;
% struct_bt_MCMC.fv.mean = sum((struct_bt_MCMC.fv.mean-bt_tot).^2,2)/nrj_tot+err_fix;

bt_MCMC = sum(( bsxfun(@minus, bt_MCMC, bt_tot) ).^2,2)/nrj_tot;
err_tot = sum(bt_MCMC,1);
[ err_tot_min, idx_min_err_tot] = min(err_tot,[],3);
[ bt_MCMC_min_error, idx_min_error] = min(bt_MCMC,[],3);
bt_MCMC_min_error = bt_MCMC_min_error+err_fix;
% bt_MCMC_min_error = min(bt_MCMC,[],3)+err_fix;
bt_MCMC_RMSE = mean(bt_MCMC,3)+err_fix;
clear bt_MCMC
struct_bt_MCMC.tot.var = struct_bt_MCMC.tot.var/nrj_tot;

if param.plot_EV_noise
    struct_bt_MEV_noise.tot.mean = sum((struct_bt_MEV_noise.tot.mean-bt_tot).^2,2)/nrj_tot+err_fix;
    struct_bt_MEV_noise.tot.one_realiz = sum((struct_bt_MEV_noise.tot.one_realiz-bt_tot).^2,2)/nrj_tot+err_fix;
    
    bt_forecast_MEV_noise = sum(( bsxfun(@minus, bt_forecast_MEV_noise, bt_tot) ).^2,2)/nrj_tot;
    err_tot_MEV_noise = sum(bt_forecast_MEV_noise,1);
    [ err_tot_min_MEV_noise, idx_min_err_tot_MEV_noise] = min(err_tot_MEV_noise,[],3);
    [ bt_forecast_MEV_noise_min_error, idx_min_error_MEV_noise] = min(bt_forecast_MEV_noise,[],3);
    bt_forecast_MEV_noise_min_error = bt_forecast_MEV_noise_min_error+err_fix;
    bt_forecast_MEV_noise_RMSE = mean(bt_forecast_MEV_noise,3)+err_fix;
    clear bt_forecast_MEV_noise
    struct_bt_MEV_noise.tot.var = struct_bt_MEV_noise.tot.var/nrj_tot;
end

%%

if param.nb_modes==2
    % if param.nb_modes==2 || param.nb_modes > 16
    figure1=figure(1);
    close(figure1)
    switch param.type_data
        case {'turb2D_blocks_truncated'}
            figure('Units','inches', ...
                'Position',[X0(1) X0(2) 6*width 4*height], ...
                'PaperPositionMode','auto');
        case 'incompact3d_wake_episode3_cut'
            figure('Units','inches', ...
                'Position',[X0(1) X0(2) width height], ...
                'PaperPositionMode','auto');
        case {'incompact3d_wake_episode3_cut_truncated', ...
                'incompact3d_wake_episode3_cut_test_basis'}
            figure('Units','inches', ...
                'Position',[X0(1) X0(2) 6*width 4*height], ...
                'PaperPositionMode','auto');
            %             figure('Units','inches', ...
            %                 'Position',[X0(1) X0(2) 6*width 4*height], ...
            %                 'PaperPositionMode','auto');
            % %             figure('Units','inches', ...
            % %                 'Position',[X0(1) X0(2) 3*width 2*height], ...
            % %                 'PaperPositionMode','auto');
        case {'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_truncated'}
            figure('Units','inches', ...
                'Position',[X0(1) X0(2) 4*width 4*height], ...
                'PaperPositionMode','auto');
            %                 'Position',[X0(1) X0(2) 4*width 4*height], ...
            %                 'PaperPositionMode','auto');
            %             %             figure('Units','inches', ...
            %             %                 'Position',[X0(1) X0(2) 6*width 4*height], ...
            %             %                 'PaperPositionMode','auto');
            %             % %             figure('Units','inches', ...
            %             % %                 'Position',[X0(1) X0(2) 3*width 2*height], ...
            %             % %                 'PaperPositionMode','auto');
        case {'DNS100_inc3d_2D_2018_11_16_blocks_truncated'}
            figure('Units','inches', ...
                'Position',[X0(1) X0(2) 4*width 4*height], ...
                'PaperPositionMode','auto');
            %             figure('Units','inches', ...
            %                 'Position',[X0(1) X0(2) 6*width 4*height], ...
            %                 'PaperPositionMode','auto');
            % %             figure('Units','inches', ...
            % %                 'Position',[X0(1) X0(2) 3*width 2*height], ...
            % %                 'PaperPositionMode','auto');
        case {'LES_3D_tot_sub_sample_blurred','inc3D_Re3900_blocks', ...
                'inc3D_Re3900_blocks119'}
            figure('Units','inches', ...
                'Position',[X0(1) X0(2) 6*width 8*height], ...
                'PaperPositionMode','auto');
            %         figure('Units','inches', ...
            %             'Position',[X0(1) X0(2) 4*width 4*height], ...
            %             'PaperPositionMode','auto');
            %         %         figure('Units','inches', ...
            %         %             'Position',[X0(1) X0(2) 4*width 2*height], ...
            %         %             'PaperPositionMode','auto');
            %         % %             'Position',[X0(1) X0(2) 2*width 4*height], ...
            %         % %             'PaperPositionMode','auto');
        case 'inc3D_Re3900_blocks_truncated'
            figure('Units','inches', ...
                'Position',[X0(1) X0(2) 4*width 4*height], ...
                'PaperPositionMode','auto');
            %             figure('Units','inches', ...
            %                 'Position',[X0(1) X0(2) 3*width 2*height], ...
            %                 'PaperPositionMode','auto');
        otherwise
            figure('Units','inches', ...
                'Position',[X0(1) X0(2) 4*width 4*height], ...
                'PaperPositionMode','auto');
    end
    %     figure('Units','inches', ...
    %         'Position',[X0(1) X0(2) 4*width 2*height], ...
    %         'PaperPositionMode','auto');
else
    figure(1);
    %     figure(param.nb_modes + 1);
end
%  if param.nb_modes==4 && strcmp(param.type_data, 'inc3D_Re3900_blocks119')
%       figure('Units','inches', ...
%             'Position',[X0(1) X0(2) 4*width 2*height], ...
%             'PaperPositionMode','auto');
%  end

if strcmp(param.type_data, 'LES_3D_tot_sub_sample_blurred') ...
        || strcmp(param.type_data, 'inc3D_Re3900_blocks') ...
        || strcmp(param.type_data, 'inc3D_Re3900_blocks119')
    subplot(4,4,log2(param.nb_modes));
    %     subplot(4,4,param.nb_modes/2);
    % %     subplot(2,4,param.nb_modes/2);
    % % %     subplot(4,2,param.nb_modes/2);
elseif strcmp(param.type_data, 'inc3D_Re3900_blocks_truncated')...
        || strcmp(param.type_data, 'turb2D_blocks_truncated')
    %     if param.nb_modes<= 16
    %         subplot(2,2,log2(param.nb_modes));
    %     end
    subplot(2,3,log2(param.nb_modes));
    
    
    %     subplot(2,3,log2(param.nb_modes));
    % %     subplot(4,4,param.nb_modes/2);
    % % %     subplot(2,4,param.nb_modes/2);
    % % % %     subplot(4,2,param.nb_modes/2);
    
    % elseif ( strcmp(param.type_data, 'incompact3d_wake_episode3_cut_truncated')  ) ...
    %         && (param.nb_modes<=16)
    %     subplot(2,3,log2(param.nb_modes));
    % %     if log2(param.nb_modes) == 3
    % %         subplot('Position',[width/4/2+width/4/5 height/4*2/5 width/4*(1-1/5) height/4*(1-1/5)]);
    % % %         subplot(2,2,0.51+log2(param.nb_modes));
    % % %         subplot(2,2,[0 1 ]+log2(param.nb_modes));
    % %     end
elseif strcmp(param.type_data, 'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_truncated')
    subplot(2,2,param.nb_modes/2);
% subplot(2,2,log2(param.nb_modes));
    %     subplot(3,2,log2(param.nb_modes));
    % % elseif ( strcmp(param.type_data, 'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_truncated') ) ...
    % %         && (param.nb_modes<=16)
    % %     subplot(2,2,log2(param.nb_modes));
    % % %     if log2(param.nb_modes) == 3
    % % %         subplot('Position',[width/4/2+width/4/5 height/4*2/5 width/4*(1-1/5) height/4*(1-1/5)]);
    % % % %         subplot(2,2,0.51+log2(param.nb_modes));
    % % % %         subplot(2,2,[0 1 ]+log2(param.nb_modes));
    % % %     end
elseif ( strcmp(param.type_data, 'DNS100_inc3d_2D_2018_11_16_blocks_truncated') ) ...
        && (param.nb_modes<=16)
        subplot(2,2,(param.nb_modes)/2);
%     subplot(2,2,log2(param.nb_modes));
    %     if log2(param.nb_modes) == 3
    %         subplot('Position',[width/4/2+width/4/5 height/4*2/5 width/4*(1-1/5) height/4*(1-1/5)]);
    % %         subplot(2,2,0.51+log2(param.nb_modes));
    % %         subplot(2,2,[0 1 ]+log2(param.nb_modes));
    %     end
else
    subplot(2,2,(param.nb_modes)/2);
end
k=1;

% Real values

hold on;

if param.plot_EV_noise
    delta_MEV = sqrt(abs (struct_bt_MEV_noise.tot.var(:,k)));
    h_MEV = area (time_ref, [sqrt(err_fix), ...
        delta_MEV]);
%     h_MEV = area(time_ref, [struct_bt_MEV_noise.qtl(:,k), ...
%         struct_bt_MEV_noise.diff(:,k)]);
    set (h_MEV(1), 'FaceColor', 'none');
    %         set (h_MEV(2), 'FaceColor', [0.6 0.9 0.9]);
    %         set (h_MEV(2), 'FaceColor', [0.8 0.95 0.95]);
    set (h_MEV(2), 'FaceColor', [0.85 0.95 0.95]);
    %         set (h_MEV(2), 'FaceColor', [0.9 0.975 0.975]);
    %         set (h_MEV(2), 'FaceColor', [0.6 0.8 0.8]);
    set (h_MEV, 'LineStyle', '-', 'LineWidth', 1, 'EdgeColor', 'none');
    % Raise current axis to the top layer, to prevent it
    % from being hidden by the grayed area
    set (gca, 'Layer', 'top');
    
end

delta = sqrt(abs (struct_bt_MCMC.tot.var(:,k)));
% delta = (1.96) * sqrt(abs (struct_bt_MCMC.tot.var(:,k)));
h = area (time_ref, [sqrt(err_fix), ...
    delta]);
%     set(h(1),'FaceColor',[0,0.25,0.25]);
%     set(h(2),'FaceColor',[0,0.5,0.5]);
set (h(1), 'FaceColor', 'none');
set (h(2), 'FaceColor', [0.8 0.8 0.8]);
set (h, 'LineStyle', '-', 'LineWidth', 1, 'EdgeColor', 'none');

% Raise current axis to the top layer, to prevent it
% from being hidden by the grayed area
set (gca, 'Layer', 'top');

plot(time,sqrt(bt_0),'k', 'LineWidth', LineWidth);
% plot(time,sqrt(struct_bt_MCMC.tot.one_realiz(:,k))','y', 'LineWidth', LineWidth);
plot(time,sqrt(err_fix),'--k', 'LineWidth', LineWidth);

if plot_deter
    plot(time,sqrt(bt_forecast_deter(:,k))','b', 'LineWidth', LineWidth);
end
if plot_EV
    plot(time,sqrt(bt_forecast_MEV(:,k))','b--', 'LineWidth', LineWidth);
    %     plot(time,sqrt(bt_forecast_MEV(:,k))','g', 'LineWidth', LineWidth);
end
% plot(time,sqrt(bt_pseudoSto(:,k))','r--', 'LineWidth', LineWidth);

plot(time,sqrt(struct_bt_MCMC.tot.mean(:,k))','g', 'LineWidth', LineWidth);
% plot(time,sqrt(struct_bt_MCMC.fv.mean(:,k))','c', 'LineWidth', LineWidth);

plot(time,sqrt(bt_MCMC_RMSE(:,k))','r', 'LineWidth', LineWidth);
% plot(time,sqrt(bt_MCMC_RMSE(:,k))','+m', 'LineWidth', LineWidth);
plot(time,sqrt(bt_MCMC_min_error(:,k))','m', 'LineWidth', LineWidth);

if param.plot_EV_noise
    plot(time,sqrt(struct_bt_MEV_noise.tot.mean(:,k))','g-+', 'LineWidth', LineWidth);
    plot(time,sqrt(bt_forecast_MEV_noise_RMSE(:,k))','r-+', 'LineWidth', LineWidth);
    plot(time,sqrt(bt_forecast_MEV_noise_min_error(:,k))','m-+', 'LineWidth', LineWidth);
end

% if plot_modal_dt
%     plot(time,sqrt(bt_forecast_sto_a_cst_modal_dt(:,k))','or', 'LineWidth', LineWidth);
%     plot(time,sqrt(bt_forecast_sto_a_NC_modal_dt(:,k))','om', 'LineWidth', LineWidth);
% end

hold off;

% %%

if ~logscale
    err_min=0;
end

%     ax=[time(1) time(end) ...
%         max([max(abs(bt_forecast_deter(:,k))) ...
%         max(abs(bt_tot(:,k)))])*[0 1] ];
switch param.type_data
    case 'incompact3d_wake_episode3_cut'
        err_min=0.15;
        
    case 'incompact3d_wake_episode3_cut_truncated'
        err_min=0.005;
        %     err_min=0.05;
        if param.nb_modes>16
            err_min=0.005;
        end
        %     err_min=0.15;
        
        if ~logscale
            err_min=0;
        end
        
        % err_min=0.13;
        
    case 'LES_3D_tot_sub_sample_blurred'
        err_min=0.3;
    case { 'inc3D_Re3900_blocks',...
            'inc3D_Re3900_blocks119'}
        err_min=0.4;
        %     err_min=0.5;
    case 'inc3D_Re3900_blocks_truncated'
        
        %     err_min=0.005;
        
        err_min=0.4;
        %     err_min=0.45;
        % %     err_min=0.5;
    otherwise
        err_min=0;
end
ax=[time(1) time(end) err_min 1 ];
axis(ax)
if logscale
    %     set(gca,...
    %         'Units','normalized',...
    %         'FontUnits','points',...
    %         'FontWeight','normal',...
    %         'FontSize',FontSize,...
    %         'FontName','Times',...
    %         'YTick',YTick,...
    %         'YScale','log')
    set(gca,...
        'YGrid','on', ...
        'Units','normalized',...
        'FontUnits','points',...
        'FontWeight','normal',...
        'FontSize',FontSize,...
        'FontName','Times',...
        'YScale','log')
    %     set(gca,...
    %         'YGrid','on', ...
    %         'Units','normalized',...
    %         'FontUnits','points',...
    %         'FontWeight','normal',...
    %         'FontSize',FontSize,...
    %         'FontName','Times',...
    %         'YTick',YTick,...
    %         'YScale','log')
    ylabel({'error(log)'},...
        'FontUnits','points',...
        'interpreter','latex',...
        'FontSize',FontSize,...
        'FontName','Times')
else
    set(gca,...
        'YGrid','on', ...
        'Units','normalized',...
        'FontUnits','points',...
        'FontWeight','normal',...
        'FontSize',FontSize,...
        'FontName','Times')
    %         'FontName','Times',...
    %         'YTick',YTick)
    ylabel({'norm. error'},...
        'FontUnits','points',...
        'interpreter','latex',...
        'FontSize',FontSize,...
        'FontName','Times')
    
    
    if strcmp(param.type_data,'incompact3d_wake_episode3_cut_truncated')
        ax=[time(1) time(end) err_min 0.37 ];
        %         ax=[time(1) time(end) err_min 0.27 ];
    end
end
xlabel('Time',...
    'FontUnits','points',...
    'FontWeight','normal',...
    'FontSize',FontSize,...
    'FontName','Times')
title(['$n=' num2str(param.nb_modes) '$'],...
    'FontUnits','points',...
    'FontWeight','normal',...
    'interpreter','latex',...
    'FontSize',FontSizeTtitle,...
    'FontName','Times')

axis(ax)

%%

if logscale && strcmp(param.type_data,'inc3D_Re3900_blocks_truncated')
    
    
    %     err_min=0.005;
    
    err_min=0.45;
    %     err_min=0.5;
    
    YTick=0.4:0.1:1;
    % YTick=[ax(3) sqrt([err_fix(1) (1-nrj_mean/nrj_tot) 1])];
    
    %     set(gca,...
    %         'YGrid','on', ...
    %         'Units','normalized',...
    %         'FontUnits','points',...
    %         'FontWeight','normal',...
    %         'FontSize',FontSize,...
    %         'FontName','Times',...
    %         'XTickMode','auto',...
    %         'YScale','log')
    set(gca,...
        'YGrid','on', ...
        'Units','normalized',...
        'FontUnits','points',...
        'FontWeight','normal',...
        'FontSize',FontSize,...
        'FontName','Times',...
        'YTick',YTick,...
        'YScale','log')
end
% ax=[time(1) time(end) err_min 1 ];
% axis(ax)

%%

drawnow;

threshold = num2str(param.decor_by_subsampl.spectrum_threshold);
iii = (threshold =='.');
threshold(iii)='_';



% warning('Coefficients modified to study sensibility');
% param.folder_results = [param.folder_results 'sensibility_x_' ...
%     num2str(param.coef_sensitivity) '/'];
%
% % eval( ['print -depsc ' param.folder_results 'sum_modes_n=' num2str(param.nb_modes) '.eps']);
% if isfield(param,'test_basis') && param.test_basis
%     eval( ['print -depsc ' param.folder_results ...
%         param.type_data '_' ...
%         'sum_modes_n=' num2str(param.nb_modes) ...
%         '_trshld' threshold ...
%         '_test_basis.eps']);
%
% else
%     eval( ['print -depsc ' param.folder_results ...
%         param.type_data '_' ...
%         'sum_modes_n=' num2str(param.nb_modes) ...
%         '_trshld' threshold '.eps']);
% end
% % eval( ['print -depsc ' param.folder_results ...
% %     'sum_modes_n=' num2str(param.nb_modes) ...
% %     '_a_t' num2str(param.a_time_dependant) ...
% %     '_trshld' num2str(param.decor_by_subsampl.spectrum_threshold) '.eps']);

