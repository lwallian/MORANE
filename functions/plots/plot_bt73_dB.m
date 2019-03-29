function plot_bt73_dB(param,bt_forecast_sto_scalar,bt_forecast_sto_beta,...
    bt_forecast_deter, ...
    bt_forecast_MEV,bt_sans_coef1,bt_sans_coef2,bt_tot)
% Plot the sum of the error along time (in log scale)
%

clear height

width=2;
height=1.6;
% width=2.5;
% height=1.6;
% % width=1.5;
% % height=1.2;

% width=2;
% % width=1.5;
% height=1.5;
% % height=2.2;
X0=[0 0];
% 2 et 1.8

plot_deter=param.plot.plot_deter;
plot_EV=param.plot.plot_EV;
plot_tuned=param.plot.plot_tuned;

param.nb_modes = size(bt_tot,2);
param.type_data
dt_tot=param.dt;

N_time_final=param.N_tot;
N_test=param.N_test;
nb_modes=param.nb_modes;
time=(max(1,N_time_final-2*N_test+1):N_time_final)*dt_tot;

%%
nrj_varying_tot=18.2705;
load resultats/nrj_mean_LES3900;
nrj_mean = norm0; clear norm0;
nrj_tot=nrj_mean+nrj_varying_tot;
err_fix =  (nrj_varying_tot - sum(param.lambda))/nrj_tot;
% err_fix =  1- (nrj_mean+sum(param.lambda))/nrj_tot;
bt_forecast_deter = sum((bt_forecast_deter-bt_tot).^2,2)/nrj_tot+err_fix;
bt_forecast_sto_scalar = sum((bt_forecast_sto_scalar-bt_tot).^2,2)/nrj_tot+err_fix;
bt_forecast_sto_beta = sum((bt_forecast_sto_beta-bt_tot).^2,2)/nrj_tot+err_fix;
bt_sans_coef1 = sum((bt_sans_coef1-bt_tot).^2,2)/nrj_tot+err_fix;
bt_sans_coef2 = sum((bt_sans_coef2-bt_tot).^2,2)/nrj_tot+err_fix;
bt_forecast_MEV = sum((bt_forecast_MEV-bt_tot).^2,2)/nrj_tot+err_fix;

%%

if nb_modes==2
    figure('Units','inches', ...
        'Position',[X0(1) X0(2) 2*width 4*height], ...
        'PaperPositionMode','auto');
    %     figure('Units','inches', ...
    %         'Position',[X0(1) X0(2) 4*width 2*height], ...
    %         'PaperPositionMode','auto');
end
subplot(4,2,nb_modes/2);

k=1;

% Real values

hold on;
if plot_EV
    plot(time,sqrt(bt_forecast_MEV(:,k))','g');
end
plot(time,sqrt(bt_sans_coef1(:,k))','r');
plot(time,sqrt(bt_sans_coef2(:,k))','m');
if plot_tuned
    plot(time,sqrt(bt_forecast_sto_scalar(:,k))','c');
    plot(time,sqrt(bt_forecast_sto_beta(:,k))','y');
end
plot([time(1) time(end)],[1 1],'k');
plot([time(1) time(end)],[1 1]*sqrt((1-nrj_mean/nrj_tot)),'k');
plot([time(1) time(end)],[1 1]*sqrt(err_fix),'--k');
if plot_deter
    semilogy(time,sqrt(bt_forecast_deter(:,k))','b');
end

hold off;

%%

%     ax=[time(1) time(end) ...
%         max([max(abs(bt_forecast_deter(:,k))) ...
%         max(abs(bt_tot(:,k)))])*[0 1] ];

ax=[time(1) time(end) 0.3 1 ];
%     ax=[time(1) time(end) sqrt(0.09) 1 ];
% %     ax=[time(1) time(end) 0 40 ];
YTick=[ax(3) sqrt([err_fix (1-nrj_mean/nrj_tot) 1])];
%     YTick=[ax(3) sqrt([err_fix bt_sans_coef2(end) (1-nrj_mean/nrj_tot) 1])];
%     ax=[time(1) time(end) max(abs(bt_tot(:,k)))*[-1 1] ];
%     ax=axis;
%     ax([3 4])=max(abs(bt_tot(:,k)))*[-1 1];
%             'Position',[.15 .2 .75 .7],...
axis(ax)
set(gca,...
    'Units','normalized',...
    'FontUnits','points',...
    'FontWeight','normal',...
    'FontSize',9,...
    'FontName','Times',...
    'YTick',YTick,...
    'YScale','log')
ylabel({'error(log)'},...
    'FontUnits','points',...
    'interpreter','latex',...
    'FontSize',9,...
    'FontName','Times')
xlabel('Time',...
    'FontUnits','points',...
    'FontWeight','normal',...
    'FontSize',9,...
    'FontName','Times')
title(['$n=' num2str(nb_modes) '$'],...
    'FontUnits','points',...
    'FontWeight','normal',...
    'interpreter','latex',...
    'FontSize',10,...
    'FontName','Times')

axis(ax)
% eval( ['print -depsc ' param.folder_results 'sum_modes_n=' num2str(nb_modes) '.eps']);
% drawnow;

