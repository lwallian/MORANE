function plot_bt5(param,bt_forecast_sto,bt_forecast_deter,bt_tot)
%% Plot the first coefficients bt along time

clear height

beamer=true;

if beamer
%     % 2D
%     width=2.5;
%     height=1;
    
    % 3D
    width=4.72;
    height=3.78;
else
    width=1.5;
    height=1.2;
end

% width=2;
% % width=1.5;
% height=1.5;
% % height=2.2;
X0=[0 0];
% 2 et 1.8

plot_deter=true;

nb_modes = param.nb_modes;
param.type_data

if isfield(param,'N_tot')
    N_tot=param.N_tot;
    N_test=param.N_test;
else
    N_tot=300;
    N_test=299;
end

dt_tot=param.dt;
N_time_final=N_tot;
time=(1:(N_test+1))*dt_tot;
time_ref = time;

for k=1:nb_modes
    
    ref=max(abs(bt_tot(:,k)));
    if plot_deter
        idx=abs(bt_forecast_deter(:,k))>5*ref;
        if any(idx)
            nb_valeurs_abherantes = sum(idx)/length(idx)*100;
            ['Il y a ' num2str(nb_valeurs_abherantes) ...
                ' % de valeurs adberantes avec le modele deterministe']
           bt_forecast_deter(idx,:)=nan;
        end
    end
    idx=abs(bt_forecast_sto(:,k))>5*ref;
    if any(idx)
        nb_valeurs_abherantes = sum(idx)/length(idx)*100;;
        ['Il y a ' num2str(nb_valeurs_abherantes) ...
            ' % de valeurs adberantes avec le modele stochastique']
        bt_forecast_sto(idx,:)=nan;
    end
    %%
    figure('Units','inches',...
        'Position',[X0(1) X0(2) width height],...
        'PaperPositionMode','auto');
    %%
    
    % Real values
    plot(time_ref,bt_tot(:,k)','k.');
    hold on;
    plot(time,bt_forecast_sto(:,k)','r');
    if plot_deter
        plot(time,bt_forecast_deter(:,k)','b');
    end
    hold off;
    
%%

    ax=[time(1) time(end) ... 
        max([max(abs(bt_forecast_deter(:,k))) ... 
        max(abs(bt_tot(:,k)))])*[-1 1] ];
    YTick=ax(3):(ax(4)-ax(3))/2:0;
    YTick = [ YTick(1:end-1) -YTick(end:-1:1)];
    axis(ax)
    set(gca,...
        'Units','normalized',...
        'FontUnits','points',...
        'FontWeight','normal',...
        'FontSize',9,...
        'FontName','Times')
    ylabel({['$b_{' num2str(k) '}(t)$']},...
        'FontUnits','points',...
        'interpreter','latex',...
        'FontSize',8,...
        'FontName','Times')
    xlabel('Time',...
        'FontUnits','points',...
        'FontWeight','normal',...
        'FontSize',8,...
        'FontName','Times')
    title(['Temporal mode ' num2str(k) ],...
        'FontUnits','points',...
        'FontWeight','normal',...
        'FontSize',8,...
        'FontName','Times')
    
%%
%     axis normal
    eval( ['print -depsc ' param.folder_results num2str(nb_modes) 'm_mode' num2str(k) '.eps']);
%     eval( ['print -depsc ' param.folder_results num2str(nb_modes) 'm/mode' num2str(k) '.eps']);
    drawnow;
%     eval( ['print -depsc ' num2str(nb_modes) 'm/' ...
%         num2str(nb_modes_used) '_modes_used_for_estim_coef_correctif/mode' num2str(k) '.eps']);
%     drawnow;
end
% keyboard;
% close all
