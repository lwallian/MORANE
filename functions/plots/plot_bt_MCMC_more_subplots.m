function plot_bt_MCMC_more_subplots(param,bt_forecast_sto_scalar,bt_forecast_sto_beta,...
    bt_forecast_sto_a_cst_modal_dt, bt_forecast_sto_a_NC_modal_dt, bt_forecast_deter, ...
    bt_forecast_MEV,bt_sans_coef1,bt_sans_coef2,bt_tot,struct_bt_MCMC,bt_MCMC)
% Plot the sum of the error along time (in log scale)
%

% logscale =true
logscale =false

clear height

width=2;
height=1.5;
% width=6;
% height=2.4;
% % width=2;
% % height=1.6;
% % % width=2.5;
% % % height=1.6;
% % % % width=1.5;
% % % % height=1.2;

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
bt_MCMC=bt_MCMC(1:N_test,:,:);
struct_bt_MCMC.tot.mean=struct_bt_MCMC.tot.mean(1:N_test,:);
struct_bt_MCMC.tot.var=struct_bt_MCMC.tot.var(1:N_test,:);
struct_bt_MCMC.tot.one_realiz=struct_bt_MCMC.tot.one_realiz(1:N_test,:);
struct_bt_MCMC.fv.mean=struct_bt_MCMC.fv.mean(1:N_test,:);
struct_bt_MCMC.fv.var=struct_bt_MCMC.fv.var(1:N_test,:);
struct_bt_MCMC.fv.one_realiz=struct_bt_MCMC.fv.one_realiz(1:N_test,:);
struct_bt_MCMC.m.mean=struct_bt_MCMC.m.mean(1:N_test,:);
struct_bt_MCMC.m.var=struct_bt_MCMC.m.var(1:N_test,:);
struct_bt_MCMC.m.one_realiz=struct_bt_MCMC.m.one_realiz(1:N_test,:);
% struct_bt_MCMC=struct_bt_MCMC(1:N_test,:);
bt_tot=bt_tot(1:N_test,:);
bt_forecast_deter=bt_forecast_deter(1:N_test,:);
bt_forecast_MEV=bt_forecast_MEV(1:N_test,:);
bt_sans_coef1=bt_sans_coef1(1:N_test,:);
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
elseif strcmp(param.type_data, 'inc3D_Re3900_blocks') ...
    || strcmp(param.type_data, 'inc3D_Re3900_blocks119')
%     warning(' the value of nrj_varying_tot is wrong');
%     nrj_varying_tot=21;
%     norm0=16;
    load resultats/nrj_mean_DNS3900blurred;
    load([ param.folder_data 'inc3D_Re3900_blocks_pre_c.mat'],'c');
%     c=c*prod(param.dX)/(param.N_tot_old*param.decor_by_subsampl.n_subsampl_decor-1);
    c=c*prod(param.dX)/(param.N_tot*param.decor_by_subsampl.n_subsampl_decor-1);
    nrj_varying_tot=trace(c);
%     keyboard;
elseif strcmp(param.type_data, 'incompact3d_wake_episode3_cut')
    param.file_nrj_mU=[ param.folder_results '_nrj_mU_' param.type_data '.mat'];
    load(param.file_nrj_mU, 'nrj_mU');
    norm0=nrj_mU;
    param.name_file_pre_c = [param.folder_data param.type_data '_pre_c'];
    load(param.name_file_pre_c,'c');
%     load(['resultats/nrj_mean_' param.type_data ' .mat']);
%     load([ param.folder_data param.type_data ' .mat'],'c');
    c=c*prod(param.dX)/(param.N_tot*param.decor_by_subsampl.n_subsampl_decor-1);
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
% elseif strcmp(param.type_data, 'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_truncated')
    % param.file_nrj_mU=[ param.folder_results '_nrj_mU_' param.type_data '0.mat'];
    % load(param.file_nrj_mU, 'nrj_mU');
    % norm0=nrj_mU;
    % param.name_file_pre_c = [param.folder_data param.type_data '_pre_c'];
    % load(param.name_file_pre_c,'c');
    % c=c*prod(param.dX)/(param.N_tot*param.decor_by_subsampl.n_subsampl_decor-1);
    % nrj_varying_tot=trace(c);
else
    % if param.data_in_blocks.bool % if data are saved in several files
        % % Get some information on how the data are saved
        % %         param_blocks=read_data_blocks(param.type_data,param.folder_data);
        % %         nb_blocks = param_blocks.data_in_blocks.nb_blocks;
        % nb_blocks = param.data_in_blocks.nb_blocks;
        % param.file_nrj_mU=[ param.folder_results '_nrj_mU_' param.type_data num2str(nb_blocks) '.mat'];
    % else
        % param.file_nrj_mU=[ param.folder_results '_nrj_mU_' param.type_data '0.mat'];
    % end
    param.file_nrj_mU=[ param.folder_results '_nrj_mU_' param.type_data '0.mat'];
    load(param.file_nrj_mU, 'nrj_mU');
    norm0=nrj_mU;
    param.name_file_pre_c = [param.folder_data param.type_data '_pre_c'];
    load(param.name_file_pre_c,'c');
    c=c*prod(param.dX)/(param.N_tot*param.decor_by_subsampl.n_subsampl_decor-1);
    nrj_varying_tot=trace(c);
end

nrj_mean = norm0; clear norm0;
nrj_tot=nrj_mean+nrj_varying_tot;
err_fix =  (nrj_varying_tot - sum(param.lambda))/nrj_tot
% %     keyboard;
% % err_fix =  1- (nrj_mean+sum(param.lambda))/nrj_tot;
% bt_forecast_sto_a_cst_modal_dt = sum((bt_forecast_sto_a_cst_modal_dt-bt_tot).^2,2)/nrj_tot+err_fix;
% bt_forecast_sto_a_NC_modal_dt = sum((bt_forecast_sto_a_NC_modal_dt-bt_tot).^2,2)/nrj_tot+err_fix;
bt_forecast_deter = sum((bt_forecast_deter-bt_tot).^2,2)/nrj_tot+err_fix;
% bt_forecast_sto_scalar = sum((bt_forecast_sto_scalar-bt_tot).^2,2)/nrj_tot+err_fix;
% bt_forecast_sto_beta = sum((bt_forecast_sto_beta-bt_tot).^2,2)/nrj_tot+err_fix;
bt_sans_coef1 = sum((bt_sans_coef1-bt_tot).^2,2)/nrj_tot+err_fix;
% bt_sans_coef_a_NC = sum((bt_sans_coef_a_NC-bt_tot).^2,2)/nrj_tot+err_fix;
bt_forecast_MEV = sum((bt_forecast_MEV-bt_tot).^2,2)/nrj_tot+err_fix;
struct_bt_MCMC.tot.mean = sum((struct_bt_MCMC.tot.mean-bt_tot).^2,2)/nrj_tot+err_fix;
struct_bt_MCMC.tot.one_realiz = sum((struct_bt_MCMC.tot.one_realiz-bt_tot).^2,2)/nrj_tot+err_fix;
struct_bt_MCMC.fv.mean = sum((struct_bt_MCMC.fv.mean-bt_tot).^2,2)/nrj_tot+err_fix;
struct_bt_MCMC.tot.var = struct_bt_MCMC.tot.var/nrj_tot;

bt_MCMC = sum(( bsxfun(@minus, bt_MCMC, bt_tot) ).^2,2)/nrj_tot;
bt_MCMC_min_error = min(bt_MCMC,[],3)+err_fix;
bt_MCMC_RMSE = mean(bt_MCMC,3)+err_fix;
clear bt_MCMC

%%

if param.nb_modes==2
    figure1=figure(1);
    close(figure1)
    if strcmp(param.type_data, 'incompact3d_wake_episode3_cut')
        figure('Units','inches', ...
            'Position',[X0(1) X0(2) 2*width 2*height], ...
            'PaperPositionMode','auto');
    end
    if strcmp(param.type_data, 'LES_3D_tot_sub_sample_blurred') ...
            || strcmp(param.type_data, 'inc3D_Re3900_blocks') ...
            || strcmp(param.type_data, 'inc3D_Re3900_blocks119')
        close all;
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
    end
    if strcmp(param.type_data, 'inc3D_Re3900_blocks_truncated')
        close all;
        figure('Units','inches', ...
            'Position',[X0(1) X0(2) 3*width 2*height], ...
            'PaperPositionMode','auto');
    end
    %     figure('Units','inches', ...
    %         'Position',[X0(1) X0(2) 4*width 2*height], ...
    %         'PaperPositionMode','auto');
else
    figure(1);
end
%  if param.nb_modes==4 && strcmp(param.type_data, 'inc3D_Re3900_blocks119')
%       figure('Units','inches', ...
%             'Position',[X0(1) X0(2) 4*width 2*height], ...
%             'PaperPositionMode','auto');
%  end

if strcmp(param.type_data, 'LES_3D_tot_sub_sample_blurred') ...
        || strcmp(param.type_data, 'inc3D_Re3900_blocks') ...
    || strcmp(param.type_data, 'inc3D_Re3900_blocks119') 
    subplot(4,4,param.nb_modes/2);
%     subplot(4,4,param.nb_modes/2);
% %     subplot(2,4,param.nb_modes/2);
% % %     subplot(4,2,param.nb_modes/2);
elseif strcmp(param.type_data, 'inc3D_Re3900_blocks_truncated') 
    subplot(2,3,log2(param.nb_modes));
%     subplot(4,4,param.nb_modes/2);
% %     subplot(2,4,param.nb_modes/2);
% % %     subplot(4,2,param.nb_modes/2);
elseif strcmp(param.type_data, 'incompact3d_wake_episode3_cut') 
    subplot(2,2,log2(param.nb_modes));
end
k=1;

% Real values

hold on;

delta = (1.96) * sqrt(abs (struct_bt_MCMC.tot.var(:,k)));
h = area (time_ref, [sqrt(err_fix)*ones(N_test+1,1), ...
    delta]);
%     set(h(1),'FaceColor',[0,0.25,0.25]);
%     set(h(2),'FaceColor',[0,0.5,0.5]);
set (h(1), 'FaceColor', 'none');
set (h(2), 'FaceColor', [0.8 0.8 0.8]);
set (h, 'LineStyle', '-', 'LineWidth', 1, 'EdgeColor', 'none');

% Raise current axis to the top layer, to prevent it
% from being hidden by the grayed area
set (gca, 'Layer', 'top');

plot(time,sqrt(struct_bt_MCMC.tot.one_realiz(:,k))','y');
if plot_deter
    plot(time,sqrt(bt_forecast_deter(:,k))','b');
end
plot(time,sqrt(bt_sans_coef1(:,k))','r--');

plot(time,sqrt(struct_bt_MCMC.tot.mean(:,k))','g');
% plot(time,sqrt(struct_bt_MCMC.fv.mean(:,k))','c');

plot(time,sqrt(bt_MCMC_min_error(:,k))','m');
plot(time,sqrt(bt_MCMC_RMSE(:,k))','+m');

if plot_EV
    plot(time,sqrt(bt_forecast_MEV(:,k))','g');
end

% if plot_modal_dt
%     plot(time,sqrt(bt_forecast_sto_a_cst_modal_dt(:,k))','or');
%     plot(time,sqrt(bt_forecast_sto_a_NC_modal_dt(:,k))','om');
% end
plot([time(1) time(end)],[1 1],'k');
plot([time(1) time(end)],[1 1]*sqrt((1-nrj_mean/nrj_tot)),'k');
plot([time(1) time(end)],[1 1]*sqrt(err_fix),'--k');
% if plot_deter
%     plot(time,sqrt(bt_forecast_deter(:,k))','b');
% end

hold off;

%%

%     ax=[time(1) time(end) ...
%         max([max(abs(bt_forecast_deter(:,k))) ...
%         max(abs(bt_tot(:,k)))])*[0 1] ];
switch param.type_data
    case 'incompact3d_wake_episode3_cut'
        err_min=0.02;
        %     err_min=0.15;
        
        % err_min=0.13;
        
    case 'LES_3D_tot_sub_sample_blurred'
        err_min=0.3;
    case {'inc3D_Re3900_blocks', 'inc3D_Re3900_blocks119', ...
            'inc3D_Re3900_blocks_truncated'}
        err_min=0.4;
        %     err_min=0.5;
    otherwise
        err_min = 0;
end
ax=[time(1) time(end) err_min 1 ];
% ax=[time(1) time(end) 0.3 1 ];
% %     ax=[time(1) time(end) sqrt(0.09) 1 ];
% % %     ax=[time(1) time(end) 0 40 ];
YTick=[ax(3) sqrt([err_fix (1-nrj_mean/nrj_tot) 1])];
%     YTick=[ax(3) sqrt([err_fix bt_sans_coef_a_NC(end) (1-nrj_mean/nrj_tot) 1])];
%     ax=[time(1) time(end) max(abs(bt_tot(:,k)))*[-1 1] ];
%     ax=axis;
%     ax([3 4])=max(abs(bt_tot(:,k)))*[-1 1];
%             'Position',[.15 .2 .75 .7],...
axis(ax)
if logscale
    set(gca,...
        'Units','normalized',...
        'FontUnits','points',...
        'FontWeight','normal',...
        'FontSize',11,...
        'FontName','Times',...
        'YTick',YTick,...
        'YScale','log')
    ylabel({'error(log)'},...
        'FontUnits','points',...
        'interpreter','latex',...
        'FontSize',11,...
        'FontName','Times')
else
    set(gca,...
        'Units','normalized',...
        'FontUnits','points',...
        'FontWeight','normal',...
        'FontSize',11,...
        'FontName','Times',...
        'YTick',YTick)
    ylabel({'error'},...
        'FontUnits','points',...
        'interpreter','latex',...
        'FontSize',11,...
        'FontName','Times')
end
xlabel('Time',...
    'FontUnits','points',...
    'FontWeight','normal',...
    'FontSize',11,...
    'FontName','Times')
title(['$n=' num2str(param.nb_modes) '$'],...
    'FontUnits','points',...
    'FontWeight','normal',...
    'interpreter','latex',...
    'FontSize',12,...
    'FontName','Times')

axis(ax)
drawnow;
% eval( ['print -depsc ' param.folder_results 'sum_modes_n=' num2str(param.nb_modes) '.eps']);
eval( ['print -depsc ' param.folder_results ...
    param.type_data '_' ...
    'sum_modes_n=' num2str(param.nb_modes) ...
    '_trshld' num2str(param.decor_by_subsampl.spectrum_threshold) '.eps']);
% eval( ['print -depsc ' param.folder_results ...
%     'sum_modes_n=' num2str(param.nb_modes) ...
%     '_a_t' num2str(param.a_time_dependant) ...
%     '_trshld' num2str(param.decor_by_subsampl.spectrum_threshold) '.eps']);

