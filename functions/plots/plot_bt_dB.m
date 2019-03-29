function plot_bt_dB(param,bt_forecast_EV,bt_forecast_sto_beta,...
    bt_forecast_sto_a_cst_modal_dt, bt_forecast_sto_a_NC_modal_dt, bt_forecast_deter, ...
    bt_forecast_MEV,bt_sans_coef_a_cst,bt_sans_coef_a_NC,bt_tot)
% Plot the sum of the error along time (in log scale)
%

taille_police = 12;

clear height

width=4;
height=2.4;

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

param.nb_modes = size(bt_tot,2);
param.type_data
dt_tot=param.dt;

N_time_final=param.N_tot;
N_test=param.N_test;
nb_modes=param.nb_modes;
time=(0:(param.N_test))*dt_tot;
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
    c=c*prod(param.dX)/(param.N_tot_old*param.decor_by_subsampl.n_subsampl_decor-1);
    %     c=c*prod(param.dX)/(param.N_tot*param.decor_by_subsampl.n_subsampl_decor-1);
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
else
    param.file_nrj_mU=[ param.folder_results '_nrj_mU_' param.type_data '0.mat'];
    load(param.file_nrj_mU, 'nrj_mU');
    norm0=nrj_mU;
    param.name_file_pre_c = [param.folder_data param.type_data '_pre_c'];
    load(param.name_file_pre_c,'c');
    %     load(['resultats/nrj_mean_' param.type_data ' .mat']);
    %     load([ param.folder_data param.type_data ' .mat'],'c');
    c=c*prod(param.dX)/(param.N_tot*param.decor_by_subsampl.n_subsampl_decor-1);
    nrj_varying_tot=trace(c);
    %     keyboard;
    %     warning('no values for the total energy and the energy of the mean')
    %     norm0=0;
    %     nrj_varying_tot=0;
end

nrj_mean = norm0; clear norm0;
nrj_tot=nrj_mean+nrj_varying_tot;
err_fix =  (nrj_varying_tot - sum(param.lambda))/nrj_tot
%     keyboard;
% err_fix =  1- (nrj_mean+sum(param.lambda))/nrj_tot;
bt_forecast_sto_a_cst_modal_dt = sum((bt_forecast_sto_a_cst_modal_dt-bt_tot).^2,2)/nrj_tot+err_fix;
bt_forecast_sto_a_NC_modal_dt = sum((bt_forecast_sto_a_NC_modal_dt-bt_tot).^2,2)/nrj_tot+err_fix;
bt_forecast_deter = sum((bt_forecast_deter-bt_tot).^2,2)/nrj_tot+err_fix;
bt_forecast_EV = sum((bt_forecast_EV-bt_tot).^2,2)/nrj_tot+err_fix;
bt_forecast_sto_beta = sum((bt_forecast_sto_beta-bt_tot).^2,2)/nrj_tot+err_fix;
bt_sans_coef_a_cst = sum((bt_sans_coef_a_cst-bt_tot).^2,2)/nrj_tot+err_fix;
bt_sans_coef_a_NC = sum((bt_sans_coef_a_NC-bt_tot).^2,2)/nrj_tot+err_fix;
bt_forecast_MEV = sum((bt_forecast_MEV-bt_tot).^2,2)/nrj_tot+err_fix;

%%

if nb_modes==2
    switch param.type_data
        case 'incompact3d_wake_episode3_cut'
            figure('Units','inches', ...
                'Position',[X0(1) X0(2) width height], ...
                'PaperPositionMode','auto');
        case 'incompact3d_wake_episode3_cut'
            figure('Units','inches', ...
                'Position',[X0(1) X0(2) width height], ...
                'PaperPositionMode','auto');
        case {'LES_3D_tot_sub_sample_blurred', ...
                'inc3D_Re3900_blocks', ...
                'inc3D_Re3900_blocks119'}
            figure('Units','inches', ...
                'Position',[X0(1) X0(2) 4*width 2*height], ...
                'PaperPositionMode','auto');
            %             'Position',[X0(1) X0(2) 2*width 4*height], ...
            %             'PaperPositionMode','auto');
        case 'inc3D_HRLESlong_Re3900_blocks'
            figure('Units','inches', ...
                'Position',[X0(1) X0(2) 2*width 2*height], ...
                'PaperPositionMode','auto');
        case {'DNS300_inc3d_3D_2017_04_02_blocks', ...
                'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks', ...
                'DNS300_inc3d_3D_2017_04_09_NOT_BLURRED_blocks', ...
                'DNS100_inc3d_2D_2017_04_29_blocks', ...
                'DNS300_inc3d_3D_2017_04_09_blocks'}
            figure('Units','inches', ...
                'Position',[X0(1) X0(2) 2*width height], ...
                'PaperPositionMode','auto');
            
    end
end
if nb_modes==4 && strcmp(param.type_data, 'inc3D_Re3900_blocks119')
    figure('Units','inches', ...
        'Position',[X0(1) X0(2) 4*width 2*height], ...
        'PaperPositionMode','auto');
end


switch param.type_data
    case  {'DNS300_inc3d_3D_2017_04_02_blocks', ...
            'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks', ...
            'DNS300_inc3d_3D_2017_04_09_NOT_BLURRED_blocks', ...
            'DNS300_inc3d_3D_2017_04_09_blocks'}
        subplot(1,2,log(nb_modes)/log(2));
    case  'inc3D_HRLESlong_Re3900_blocks'
        subplot(2,2,log(nb_modes)/log(2));
        time(end)=[];
    case{ 'LES_3D_tot_sub_sample_blurred',...
            'inc3D_Re3900_blocks', ...
            'inc3D_Re3900_blocks119'}
        subplot(2,4,nb_modes/2);
        %     subplot(4,2,nb_modes/2);
end
k=1;

% Real values

hold on;
if plot_EV
    plot(time,sqrt(bt_forecast_MEV(:,k))','--g');
    plot(time,sqrt(bt_forecast_EV(:,k))','g');
end
plot(time,sqrt(bt_sans_coef_a_cst(:,k))','r');
switch param.type_data
    case {'LES_3D_tot_sub_sample_blurred', ...
            'inc3D_Re3900_blocks', ...
            'inc3D_Re3900_blocks119', ...
            'DNS300_inc3d_3D_2017_04_02_blocks', ...
            'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks', ...
            'DNS300_inc3d_3D_2017_04_09_NOT_BLURRED_blocks', ...
            'DNS100_inc3d_2D_2017_04_29_blocks', ...
            'DNS300_inc3d_3D_2017_04_09_blocks'}
        plot(time,sqrt(bt_sans_coef_a_NC(:,k))','m');
end
if plot_tuned
    plot(time,sqrt(bt_forecast_sto_scalar(:,k))','c');
    plot(time,sqrt(bt_forecast_sto_beta(:,k))','y');
end
if plot_modal_dt
    plot(time,sqrt(bt_forecast_sto_a_cst_modal_dt(:,k))','--r');
    plot(time,sqrt(bt_forecast_sto_a_NC_modal_dt(:,k))','--m');
end
plot([time(1) time(end)],[1 1],'k');
plot([time(1) time(end)],[1 1]*sqrt((1-nrj_mean/nrj_tot)),'k');
plot([time(1) time(end)],[1 1]*sqrt(err_fix),'--k');
if plot_deter
    plot(time,sqrt(bt_forecast_deter(:,k))','b');
end

hold off;

%%

%     ax=[time(1) time(end) ...
%         max([max(abs(bt_forecast_deter(:,k))) ...
%         max(abs(bt_tot(:,k)))])*[0 1] ];

switch param.type_data
    case 'incompact3d_wake_episode3_cut'
        err_min=0.15;
    case 'LES_3D_tot_sub_sample_blurred'
        err_min=0.3;
    case {'inc3D_Re3900_blocks', 'inc3D_Re3900_blocks119'}
        err_min=0.5;
    case 'inc3D_HRLESlong_Re3900_blocks'
        err_min=0;
        %     case {'DNS300_inc3d_3D_2017_04_09_NOT_BLURRED_blocks',...
        %             'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks',...
        %             'DNS300_inc3d_3D_2017_04_09_blocks',...
        %             'DNS300_inc3d_3D_2017_04_02_blocks'}
    otherwise
        err_min=0;
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

if isfield(param.coef_correctif_estim,'learning_time')
    hold on
    N_estim = param.N_learn_coef_a;
    plot(  N_estim*dt_tot*[ 1 1],ax([3 4]),'k');
    hold off
end


axis(ax)

if strcmp(param.type_data, 'inc3D_HRLESlong_Re3900_blocks')
    XTick = [0 5 10 15];
    set(gca,...
        'Units','normalized',...
        'FontUnits','points',...
        'FontWeight','normal',...
        'FontSize',taille_police,...
        'FontName','Times',...
        'YGrid','on', ...
        'XTick',XTick,...
        'YScale','log')
else
    set(gca,...
        'Units','normalized',...
        'FontUnits','points',...
        'FontWeight','normal',...
        'FontSize',taille_police,...
        'FontName','Times',...
        'YGrid','on', ...
        'YScale','log')
end
% set(gca,...
%     'Units','normalized',...
%     'FontUnits','points',...
%     'FontWeight','normal',...
%     'FontSize',taille_police,...
%     'FontName','Times',...
%     'YTick',YTick,...
%     'YScale','log')
ylabel({'error(log)'},...
    'FontUnits','points',...
    'interpreter','latex',...
    'FontSize',taille_police,...
    'FontName','Times')
xlabel('$t$',...
    'FontUnits','points',...
    'FontWeight','normal',...
    'interpreter','latex',...
    'FontSize',taille_police,...
    'FontName','Times')
% title(['$n=' num2str(nb_modes) '$'],...
%     'FontUnits','points',...
%     'FontWeight','normal',...
%     'interpreter','latex',...
%     'FontSize',12,...
%     'FontName','Times')

axis(ax)
drawnow;
% % eval( ['print -depsc ' param.folder_results 'sum_modes_n=' num2str(nb_modes) '.eps']);
% eval( ['print -depsc ' param.folder_results ...
%     'sum_modes_n=' num2str(nb_modes) ...
%     '_trshld' num2str(param.decor_by_subsampl.spectrum_threshold) '.eps']);
% % eval( ['print -depsc ' param.folder_results ...
% %     'sum_modes_n=' num2str(nb_modes) ...
% %     '_a_t' num2str(param.a_time_dependant) ...
% %     '_trshld' num2str(param.decor_by_subsampl.spectrum_threshold) '.eps']);

