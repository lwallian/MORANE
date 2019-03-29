function plot_bt(param,bt_forecast_EV,bt_forecast_sto_beta,...
    bt_forecast_sto_a_cst_modal_dt, bt_forecast_sto_a_NC_modal_dt, bt_forecast_deter,...
    bt_forecast_MEV,bt_sans_coef_a_cst,bt_sans_coef_a_NC,bt_tot)
% function plot_bt(param,bt_forecast_EV,bt_forecast_sto_beta,...
%     bt_forecast_sto_a_cst_modal_dt, bt_forecast_sto_a_NC_modal_dt, bt_forecast_deter, ...
%     bt_forecast_MEV,bt_sans_coef_a_cst,bt_sans_coef_a_NC,bt_tot)
%% Plot the first coefficients bt along time


MarkerSize = 3;

beamer=true;

plot_deter=param.plot.plot_deter;
plot_EV=param.plot.plot_EV;
plot_tuned=param.plot.plot_tuned;
plot_modal_dt=param.plot_modal_dt;

if beamer
    % 2D
    width=2.5;
    height=2;
    %     height=1;
    
    %     % 3D
    %     width=4.72;
    %     height=3.78;
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
% N_time_final=N_tot;
%time=(1:N_tot)*dt_tot;
time=(0:(N_test-1))*dt_tot;
time_ref = time;


%%
%height=height/1.5;

if strcmp(param.type_data, 'DNS100_inc3d_2D_2017_04_29_blocks')
time=(0:(N_test))*dt_tot;
time_ref = time;

    width=width*2;
    param.N_test = 50;
    time = time(end-param.N_test:end);
    bt_forecast_deter = bt_forecast_deter(end-param.N_test:end,:);
    bt_forecast_MEV = bt_forecast_MEV(end-param.N_test:end,:);
    bt_forecast_EV = bt_forecast_EV(end-param.N_test:end,:);
    bt_forecast_sto_beta = bt_forecast_sto_beta(end-param.N_test:end,:);
    bt_forecast_sto_a_cst_modal_dt = bt_forecast_sto_a_cst_modal_dt(end-param.N_test:end,:);
    bt_forecast_sto_a_NC_modal_dt = bt_forecast_sto_a_NC_modal_dt(end-param.N_test:end,:);
    bt_sans_coef_a_cst = bt_sans_coef_a_cst(end-param.N_test:end,:);
    % elseif strcmp(param.type_data, 'inc3D_HRLESlong_Re3900_blocks')
    %     bt_forecast_EV = bt_forecast_MEV;
    
    plot_modal_dt=false;
    plot_not_modal_dt=true;
    plot_a_cst=true;
    plot_a_NC=false;
else
    height=height/2;
    plot_modal_dt=true;
    plot_not_modal_dt=false;
    plot_a_cst=false;
    plot_a_NC=true;
end

%%


for k=1:nb_modes
    
    %     ref=max(abs(bt_tot(:,k)));
    %     if plot_deter
    %         idx=abs(bt_forecast_deter(:,k))>5*ref;
    %         if any(idx)
    %             nb_valeurs_abherantes = sum(idx)/length(idx)*100;
    %             ['Il y a ' num2str(nb_valeurs_abherantes) ...
    %                 ' % de valeurs adberantes avec le modele deterministe']
    %            bt_forecast_deter(idx,:)=nan;
    %         end
    %     end
    %     idx=abs(bt_forecast_sto(:,k))>5*ref;
    %     if any(idx)
    %         nb_valeurs_abherantes = sum(idx)/length(idx)*100;;
    %         ['Il y a ' num2str(nb_valeurs_abherantes) ...
    %             ' % de valeurs adberantes avec le modele stochastique']
    %         bt_forecast_sto(idx,:)=nan;
    %     end
    %%
    figure('Units','inches',...
        'Position',[X0(1) X0(2) width height],...
        'PaperPositionMode','auto');
    %%
    
    %     % Real values
    %     plot(time_ref,bt_tot(:,k)','k.');
    %     hold on;
    %     plot(time,bt_forecast_sto(:,k)','r');
    %     if plot_deter
    %         plot(time,bt_forecast_deter(:,k)','b');
    %     end
    %     hold off;
    %%
    
    hold on;
    % Real values
    plot(time_ref,bt_tot(:,k)','ko--',...
            'MarkerSize',MarkerSize);
    %     plot(time_ref,bt_tot(:,k)','k.');
    if plot_EV
        if plot_modal_dt
            plot(time, (bt_forecast_MEV(:,k))','--g');
        end
        if plot_not_modal_dt
            plot(time, (bt_forecast_EV(:,k))','g');
        end
    end
    if plot_not_modal_dt
        if plot_a_cst
            plot(time, (bt_sans_coef_a_cst(:,k))','r');
        end
        if plot_a_NC
            plot(time, (bt_sans_coef_a_NC(:,k))','m');
        end
    end
    if plot_tuned
        plot(time, (bt_forecast_sto_scalar(:,k))','c');
        plot(time, (bt_forecast_sto_beta(:,k))','y');
    end
    if plot_modal_dt
        if plot_a_cst
            plot(time, (bt_forecast_sto_a_cst_modal_dt(:,k))','--r');
        end
        if plot_a_NC
            plot(time, (bt_forecast_sto_a_NC_modal_dt(:,k))','--m');
        end
    end
    %     plot([time(1) time(end)],[1 1],'k');
    %     plot([time(1) time(end)],[1 1]* ((1-nrj_mean/nrj_tot)),'k');
    %     plot([time(1) time(end)],[1 1]* (err_fix),'--k');
    if plot_deter
        plot(time, (bt_forecast_deter(:,k))','b');
        %     semilogy(time, (bt_forecast_deter(:,k))','b');
    end
    
    hold off;
    
    %%
    
    %     ax=[time(1) time(end) ...
    %         max([max(abs(bt_forecast_deter(:,k))) ...
    %         max(abs(bt_tot(:,k)))])*[-1 1] ];
    ax=[time(1) time(end) ...
        max(abs(bt_tot(:,k)))*1.5*[-1 1] ];
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
    xlabel('$t$',...
        'interpreter','latex',...
        'FontUnits','points',...
        'FontWeight','normal',...
        'FontSize',8,...
        'FontName','Times')
    %     title(['Temporal mode ' num2str(k) ],...
    %         'FontUnits','points',...
    %         'FontWeight','normal',...
    %         'FontSize',8,...
    %         'FontName','Times')
    
    %%
    drawnow;
    %     axis normal
    eval( ['print -depsc ' param.folder_results num2str(nb_modes) 'm_mode' num2str(k) '.eps']);
    %     eval( ['print -depsc ' param.folder_results num2str(nb_modes) 'm/mode' num2str(k) '.eps']);
    
    %     eval( ['print -depsc ' num2str(nb_modes) 'm/' ...
    %         num2str(nb_modes_used) '_modes_used_for_estim_coef_correctif/mode' num2str(k) '.eps']);
    %     drawnow;
end
% keyboard;
% close all
