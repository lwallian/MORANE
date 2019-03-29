function plot_bt72_dB_simple(param,bt_forecast_sto_scalar,bt_forecast_sto_beta,...
        bt_forecast_deter, ...
    bt_forecast_MEV,bt_sans_coef2,bt_tot)
% function plot_bt7_dB(param,bt_forecast_sto,bt_forecast_deter,bt_forecast_MEV,bt_sans_coef,bt_tot)
%% Plot the first coefficients bt along time

clear height

width=2.5;
height=1.6;
% width=1.5;
% height=1.2;

% width=2;
% % width=1.5;
% height=1.5;
% % height=2.2;
X0=[0 0];
% 2 et 1.8

plot_deter=true;

param.nb_modes = size(bt_tot,2)
param.type_data
dt_tot=param.dt;

% nb_modes_used=param.coef_correctif_estim.nb_modes_used;

% nb_modes = min(10,size(bt,2));
N_time_final=param.N_tot;
N_test=param.N_test;
nb_modes=param.nb_modes;
% N_time_final=ceil(3*N_test/4);
time=(max(1,N_time_final-2*N_test+1):N_time_final)*dt_tot;
time_ref = time;
% bt_tot(1:(end-2*N_test),:)=[];
% bt_forecast_sto(1:(end-2*N_test),:)=[];
% bt_forecast_deter(1:(end-2*N_test),:)=[];

%%
nrj_varying_tot=18.2705;
load nrj_mean_LES3900;
nrj_mean = norm0; clear norm0;
nrj_tot=nrj_mean+nrj_varying_tot;
err_fix =  (nrj_varying_tot - sum(param.lambda))/nrj_tot;
% err_fix =  1- (nrj_mean+sum(param.lambda))/nrj_tot;
bt_forecast_deter = sum((bt_forecast_deter-bt_tot).^2,2)/nrj_tot+err_fix;
bt_forecast_sto_scalar = sum((bt_forecast_sto_scalar-bt_tot).^2,2)/nrj_tot+err_fix;
bt_forecast_sto_beta = sum((bt_forecast_sto_beta-bt_tot).^2,2)/nrj_tot+err_fix;
% bt_sans_coef1 = sum((bt_sans_coef1-bt_tot).^2,2)/nrj_tot+err_fix;
bt_sans_coef2 = sum((bt_sans_coef2-bt_tot).^2,2)/nrj_tot+err_fix;
bt_forecast_MEV = sum((bt_forecast_MEV-bt_tot).^2,2)/nrj_tot+err_fix;

%% Cut
% N=size(bt_tot,1);
% N_keep=ceil(N/8.5);
% time_ref=time_ref(end-N_keep:end);
% time=time(end-N_keep:end);
% bt_tot=bt_tot(end-N_keep:end,:);
% bt_forecast_sto=bt_forecast_sto(end-N_keep:end,:);
% bt_forecast_deter=bt_forecast_deter(end-N_keep:end,:);

%%

if nb_modes==2
    figure('Units','inches', ...
        'Position',[X0(1) X0(2) 4*width 2*height], ...
        'PaperPositionMode','auto');
end
subplot(2,4,nb_modes/2);

 k=1;
    
    %     pause;
    
    %%
    
%     ref=max(abs(bt_tot(:,k)));
%     if plot_deter
%         idx=abs(bt_forecast_deter(:,k))>5*ref;
%         if any(idx)
%             nb_valeurs_abherantes = sum(idx)/length(idx)*100;
%             ['Il y a ' num2str(nb_valeurs_abherantes) ...
%                 ' % de valeurs adberantes avec le modele deterministe']
%             idx =[find(~idx)]';
%             %             bt_tot=bt_tot(idx,:);
%             bt_forecast_deter=bt_forecast_deter(idx,:);
%             bt_forecast_sto=bt_forecast_sto(idx,:);
%             time=time(idx);
%         end
%     end
%     idx=abs(bt_forecast_sto(:,k))>5*ref;
%     if any(idx)
%         nb_valeurs_abherantes = sum(idx)/length(idx)*100;;
%         ['Il y a ' num2str(nb_valeurs_abherantes) ...
%             ' % de valeurs adberantes avec le modele stochastique']
%         idx =[find(~idx)]';
%         %         bt_tot=bt_tot(idx,:);
%         bt_forecast_deter=bt_forecast_deter(idx,:);
%         bt_forecast_sto=bt_forecast_sto(idx,:);
%         time=time(idx);
%     end
    %%
    
    % Real values
    
    hold on; 
    plot(time,sqrt(bt_forecast_MEV(:,k))','g');
%     plot(time,bt_sans_coef1(:,k)','r');
    plot(time,sqrt(bt_sans_coef2(:,k))','m');
    plot(time,sqrt(bt_forecast_sto_scalar(:,k))','c');
    plot(time,sqrt(bt_forecast_sto_beta(:,k))','y');
    plot([time(1) time(end)],[1 1],'k');
    plot([time(1) time(end)],[1 1]*sqrt((1-nrj_mean/nrj_tot)),'k');
    plot([time(1) time(end)],[1 1]*sqrt(err_fix),'--k');
    if plot_deter
        semilogy(time,sqrt(bt_forecast_deter(:,k))','b');
    end
%     plot(time,bt_forecast_MEV(:,k)','g.');
%     plot(time,bt_sans_coef(:,k)','r');
%     plot(time,bt_forecast_sto(:,k)','c');
%     plot([time(1) time(end)],[1 1]*nrj_tot,'k');
%     plot([time(1) time(end)],[1 1]*err_fix,'k');
%     if plot_deter
%         plot(time,bt_forecast_deter(:,k)','b');
%     end
    
    hold off;
    
%%

%     ax=[time(1) time(end) ... 
%         max([max(abs(bt_forecast_deter(:,k))) ... 
%         max(abs(bt_tot(:,k)))])*[0 1] ];

    ax=[time(1) time(end) 0.3 1 ];
%     ax=[time(1) time(end) sqrt(0.09) 1 ];
% %     ax=[time(1) time(end) 0 40 ];
    YTick=[ax(3) sqrt([err_fix bt_sans_coef2(end) (1-nrj_mean/nrj_tot) 1])];
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
%     set(gca,...
%         'Units','normalized',...
%         'YTick',YTick,...
%         'XTick',time(1):(time(end)-time(1))/2:time(end),...
%         'FontUnits','points',...
%         'FontWeight','normal',...
%         'FontSize',9,...
%         'FontName','Times')
    ylabel({['error(log)']},...
        'FontUnits','points',...
        'interpreter','latex',...
        'FontSize',8,...
        'FontName','Times')
    xlabel('Time',...
        'FontUnits','points',...
        'FontWeight','normal',...
        'FontSize',8,...
        'FontName','Times')
    %     legend({'$y=\sin(t)$'},...
    %         'FontUnits','points',...
    %         'interpreter','latex',...
    %         'FontSize',7,...
    %         'FontName','Times',...
    %         'Location','NorthEast')
    title(['$n=' num2str(nb_modes) '$'],...
        'FontUnits','points',...
        'FontWeight','normal',...
        'interpreter','latex',...
        'FontSize',8,...
        'FontName','Times')
    
    axis(ax)
    
    %     title(['Temporal mode ' num2str(k) ]);
    
    %     title(['Real values (o) and Estimated values at the non' ...
    %               'measuring times (+) for a deterministic (red) and stochastic (blue) models']);

%%
% % %     axis normal
% %     eval( ['print -depsc plot/threshold0.01/all_' a_t '/' num2str(nb_modes) 'm.eps']);
%     eval( ['print -depsc plot/all_' a_t '/' num2str(nb_modes) 'm.eps']);
    
        eval( ['print -depsc ' param.folder_results 'sum_modes' num2str(nb_modes) '.eps']);
    drawnow;

% %     eval( ['print -depsc plot/' num2str(nb_modes) 'm/all_modes.eps']);
% %     eval( ['print -depsc plot/' num2str(nb_modes) 'm/mode' num2str(k) '.eps']);
%     drawnow;
% % %     eval( ['print -depsc ' num2str(nb_modes) 'm/' ...
% % %         num2str(nb_modes_used) '_modes_used_for_estim_coef_correctif/mode' num2str(k) '.eps']);
% % %     drawnow;

