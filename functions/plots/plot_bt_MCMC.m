function plot_bt_MCMC(param,bt_forecast_sto_scalar,bt_forecast_sto_beta,...
    bt_forecast_sto_a_cst_modal_dt, bt_forecast_sto_a_NC_modal_dt, bt_forecast_deter, ...
    bt_forecast_MEV,struct_bt_MEV_noise,bt_pseudoSto,bt_tot,struct_bt_MCMC)
%     bt_forecast_MEV,bt_sans_coef1,bt_sans_coef2,bt_tot,struct_bt_MCMC)
%% Plot the first coefficients bt along time

clear height

beamer=true;

plot_deter=param.plot.plot_deter;
plot_EV=param.plot.plot_EV;
% plot_EV=false;
plot_tuned=false;
plot_modal_dt=false;

% % if beamer
% %     % 2D
% %     width=2.5;
% %     height=2;
% % %     height=1;
% %
% % %     % 3D
% % %     width=4.72;
% % %     height=3.78;
% % else
% %     width=1.5;
% %     height=1.2;
% % end
% width=10;
% height=2;
width=5;
height=1;
color_sexy(1,:) = [0.8500, 0.3250, 0.0980];
color_sexy(2,:) = [0.9290, 0.6940, 0.1250];
color_sexy(3,:) = [0.4940, 0.1840, 0.5560];
color_sexy(4,:) = [0.4660, 0.6740, 0.1880] 	;
color_sexy(5,:) = [0.3010, 0.7450, 0.9330];
color_sexy(6,:) = [0.6350, 0.0780, 0.1840];
Color3=[0.0 0.5 0.0];


switch param.type_data
    case 'incompact3d_wake_episode3_cut_truncated'
        width=2.5;
        height=1.5;
    case 'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_truncated'
        param.N_test = ceil(40/param.dt);
        %        param.N_test = ceil(80/param.dt);
        %        param.N_test = ceil(20/param.dt);
    case 'DNS100_OpenFOAM_2D_2020_blocks'
        param.N_test = ceil(200/param.dt);
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
    % else
    %     N_tot=300;
    %     N_test=299;
end

% warning('N_test is changed');
% N_test = 100;
if param.plot_EV_noise
    struct_bt_MEV_noise.tot.mean=struct_bt_MEV_noise.tot.mean(1:N_test,:);
    struct_bt_MEV_noise.tot.var=struct_bt_MEV_noise.tot.var(1:N_test,:);
    struct_bt_MEV_noise.tot.one_realiz=struct_bt_MEV_noise.tot.one_realiz(1:N_test,:);
end
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
bt_forecast_MEV=bt_forecast_MEV(1:N_test,:);
bt_forecast_deter=bt_forecast_deter(1:N_test,:);
bt_pseudoSto=bt_pseudoSto(1:N_test,:);
struct_bt_MCMC.tot.one_realiz=struct_bt_MCMC.tot.one_realiz(1:N_test,:);
struct_bt_MCMC.qtl=struct_bt_MCMC.qtl(1:N_test,:);
struct_bt_MCMC.diff=struct_bt_MCMC.diff(1:N_test,:);
if param.plot_EV_noise
    struct_bt_MEV_noise.qtl=struct_bt_MEV_noise.qtl(1:N_test,:);
    struct_bt_MEV_noise.diff=struct_bt_MEV_noise.diff(1:N_test,:);
end
N_test=N_test-1;

dt_tot=param.dt;
N_time_final=N_tot;
time=(1:(N_test+1))*dt_tot;
time_ref = time;

width = width *(time(end)-time(1))/80;

% warning('Coefficients modified to study sensibility');
% param.folder_results = [param.folder_results 'sensibility_x_' ...
%     num2str(param.coef_sensitivity) '/'];
% mkdir(param.folder_results)


if param.plot_EV_noise
    param.folder_results = [param.folder_results 'EV/'];
end
if param.svd_pchol
    param.type_data = [param.type_data '_svd_pchol'];
end

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
    %     axis([0 10 2*param.lambda(k)*[-1 1]])
    
    if param.plot_EV_noise
        plot(time, (struct_bt_MEV_noise.tot.mean(:,k))',...
        'Color','g');
%         plot(time, (struct_bt_MEV_noise.tot.mean(:,k))','b--');
        h_MEV = area(time_ref, [struct_bt_MEV_noise.qtl(:,k), ...
            struct_bt_MEV_noise.diff(:,k)]);
        set (h_MEV(1), 'FaceColor', 'none');
%         %         set (h_MEV(2), 'FaceColor', [0.6 0.9 0.9]);
%         %         set (h_MEV(2), 'FaceColor', [0.8 0.95 0.95]);
%         set (h_MEV(2), 'FaceColor', [0.85 0.95 0.95]);
        set (h_MEV(2), 'FaceColor', [0.8 0.8 0.8]);
        %         set (h_MEV(2), 'FaceColor', [0.9 0.975 0.975]);
        %         set (h_MEV(2), 'FaceColor', [0.6 0.8 0.8]);
        set (h_MEV, 'LineStyle', '-', 'LineWidth', 1, 'EdgeColor', 'none');
        % Raise current axis to the top layer, to prevent it
        % from being hidden by the grayed area
        set (gca, 'Layer', 'top');
    end
    
    if ~param.plot_EV_noise
        h = area(time_ref, [struct_bt_MCMC.qtl(:,k), struct_bt_MCMC.diff(:,k)]);
        %%
        %  delta = 1 * sqrt (abs (struct_bt_MCMC.tot.var(:,k))); % DEFAULT
        % %  delta = 1.96 * sqrt (abs (struct_bt_MCMC.tot.var(:,k))); % DEFAULT
        %
        % % % %     h(2) = area (time_ref,  2 * delta);
        % % % %     h(1) = area (time_ref,struct_bt_MCMC.tot.mean(:,k) - delta);
        % % %     h = area (time_ref, [struct_bt_MCMC.tot.mean(:,k) - delta, ...
        % % %         struct_bt_MCMC.tot.mean(:,k) + delta]);
        % %     h = area(time_ref, [struct_bt_MCMC.qtl(:,k), struct_bt_MCMC.diff(:,k)]);
        %     h = area (time_ref, [struct_bt_MCMC.tot.mean(:,k) - delta, ...
        %         2*delta]); % DEFAULT
        % %     set(h(1),'FaceColor',[0,0.25,0.25]);
        % %     set(h(2),'FaceColor',[0,0.5,0.5]);
        
        
        % % % %     h(2) = area (time_ref,  2 * delta);
        % % % %     h(1) = area (time_ref,struct_bt_MCMC.tot.mean(:,k) - delta);
        % % %     h = area (time_ref, [struct_bt_MCMC.tot.mean(:,k) - delta, ...
        % % %         struct_bt_MCMC.tot.mean(:,k) + delta]);
        % %     h = area(time_ref, [struct_bt_MCMC.qtl(:,k), struct_bt_MCMC.diff(:,k)]);
        
        
        % h = area (time_ref, [struct_bt_MCMC.tot.mean(:,k) - delta, ...
        % 2*delta]); % DEFAULT  COMENTEI
        
        
        
        % %     set(h(1),'FaceColor',[0,0.25,0.25]);
        % %     set(h(2),'FaceColor',[0,0.5,0.5]);
        
        % set (h(1), 'FaceColor', 'none');%COMENTEI
        % set (h(2), 'FaceColor', [0.8 0.8 0.8]);%COMENTEI
        % set (h, 'LineStyle', '-', 'LineWidth', 1, 'EdgeColor', 'none');%COMENTEI
        %
        
        set (h(1), 'FaceColor', 'none');
        set (h(2), 'FaceColor', [0.8 0.8 0.8]);
        set (h, 'LineStyle', '-', 'LineWidth', 1, 'EdgeColor', 'none');
        
        
        % Raise current axis to the top layer, to prevent it
        % from being hidden by the grayed area
        set (gca, 'Layer', 'top');
        
        %     axis([0 10 2*param.lambda(k)*[-1 1]])
    end
    
    % One realization
    %     plot(time, (struct_bt_MCMC.tot.one_realiz(:,k))','y');  COMENTEI
    if plot_deter
        plot(time, (bt_forecast_deter(:,k))','b');
        %     semilogy(time, (bt_forecast_deter(:,k))','b');
    end
    
    %     plot(time, (bt_pseudoSto(:,k))','r--');
    
    % Real values
    plot(time_ref,bt_tot(:,k)','k-.');
    
    
    
    % %     plot(time_ref,struct_bt_MCMC.tot.mean(:,k)','g');  COMENTEI
    % %     plot(time_ref,struct_bt_MCMC.fv.mean(:,k)','c');
    % %     plot(time_ref,struct_bt_MCMC.tot.one_realiz(:,k)','y');
    
    % if plot_EV
    % plot(time, (bt_forecast_MEV(:,k))','b--');
    % %         plot(time, (bt_forecast_MEV(:,k))','g');
    % end
    
    if ~param.plot_EV_noise
        plot(time_ref,struct_bt_MCMC.tot.mean(:,k)',...
        'Color','g');
    end
    %     plot(time_ref,struct_bt_MCMC.fv.mean(:,k)','c');
    %     plot(time_ref,struct_bt_MCMC.tot.one_realiz(:,k)','y');
    
    
    
    %     plot(time, (bt_sans_coef2(:,k))','m');
    %     if plot_tuned
    %         plot(time, (bt_forecast_sto_scalar(:,k))','c');
    %         plot(time, (bt_forecast_sto_beta(:,k))','y');
    %     end
    if plot_modal_dt
        plot(time, (bt_forecast_sto_a_cst_modal_dt(:,k))','or');
        plot(time, (bt_forecast_sto_a_NC_modal_dt(:,k))','om');
    end
    
    %         plot(time, (bt_sans_coef(:,k))','om');
    % %     plot([time(1) time(end)],[1 1],'k');
    % %     plot([time(1) time(end)],[1 1]* ((1-nrj_mean/nrj_tot)),'k');
    % %     plot([time(1) time(end)],[1 1]* (err_fix),'--k');
    
    
    if param.plot_EV_noise
        plot(time, (struct_bt_MEV_noise.tot.mean(:,k))',...
        'Color','g');
%             'color', [0.3 0.8 0.8],'LineWidth', 2);
%         %             'color', [0.6 0.9 0.9]);
%         %         plot(time, (struct_bt_MEV_noise.tot.mean(:,k))','b--');
%         
%         %         h_MEV = area(time_ref, [struct_bt_MEV_noise.qtl(:,k), ...
%         %             struct_bt_MEV_noise.diff(:,k)]);
%         %
%         %         set (h_MEV(1), 'FaceColor', 'none');
%         %         set (h_MEV(2), 'FaceColor', [0.8 0.8 0.8]);
%         %         set (h_MEV, 'LineStyle', '-', 'LineWidth', 1, 'EdgeColor', 'none');
%         %
%         %         % Raise current axis to the top layer, to prevent it
%         %         % from being hidden by the grayed area
%         %         set (gca, 'Layer', 'top');
        plot(time, (struct_bt_MEV_noise.tot.one_realiz(:,k))','y');
    else
        plot(time, (struct_bt_MCMC.tot.one_realiz(:,k))','y');
    end
    if plot_EV
        plot(time, (bt_forecast_MEV(:,k))',...
        'Color',color_sexy(5,:),...
        'LineStyle','--' );
        %         plot(time, (bt_forecast_MEV(:,k))','g');
    end
    
    hold off;
    
    %%
    
    ax=[time(1) time(end) ...
        2*sqrt(param.lambda(k))*[-1 1] ];
    %     ax=[time(1) time(end) ...
    %         max([ max(abs(bt_tot(:,k))) ...
    %         max(abs(+delta+struct_bt_MCMC.tot.mean(:,k))) ...
    %         max(abs(-delta+struct_bt_MCMC.tot.mean(:,k))) ...
    %         ])*[-1 1] ];
    % %     ax=[time(1) time(end) ...
    % %         max([max(abs(bt_forecast_deter(:,k))) ...
    % %         max(abs(bt_tot(:,k))) ...
    % %         max(abs(+delta+struct_bt_MCMC.tot.mean(:,k))) ...
    % %         max(abs(-delta+struct_bt_MCMC.tot.mean(:,k))) ...
    % %         ])*[-1 1] ];
    % % %     ax=[time(1) time(end) ...
    % % %         max([max(abs(bt_forecast_deter(:,k))) ...
    % % %         max(abs(bt_tot(:,k)))])*[-1 1] ];
    % % % %     ax=[0 10 2*param.lambda(k)*[-1 1]];
    %%
    
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
    
    threshold = num2str(param.decor_by_subsampl.spectrum_threshold);
    iii = (threshold =='.');
    threshold(iii)='_';
    
    drawnow;
    pause(0.1)
    
    if isfield(param,'test_basis') && param.test_basis
        eval( ['print -dpng ' param.folder_results ...
            param.type_data '_' ...
            num2str(nb_modes) ...
            'm_mode' num2str(k) '_threshold_' ...
            threshold ...
            '_fullsto_test_basis.png']);
    else
        eval( ['print -dpng ' param.folder_results ...
            param.type_data '_' ...
            num2str(nb_modes) ...
            'm_mode' num2str(k) '_threshold_' ...
            threshold ...
            '_fullsto.png']);
    end
    % if isfield(param,'test_basis') && param.test_basis
    %     eval( ['print -depsc ' param.folder_results ...
    %         param.type_data '_' ...
    %         num2str(nb_modes) ...
    %         'm_mode' num2str(k) '_threshold_' ...
    %         threshold ...
    %         '_fullsto_test_basis.eps']);
    % else
    %     eval( ['print -depsc ' param.folder_results ...
    %         param.type_data '_' ...
    %         num2str(nb_modes) ...
    %         'm_mode' num2str(k) '_threshold_' ...
    %         threshold ...
    %         '_fullsto.eps']);
    % end
    %     eval( ['print -depsc ' param.folder_results num2str(nb_modes) 'm_mode' num2str(k) '_fullsto.eps']);
    % %     eval( ['print -depsc ' param.folder_results num2str(nb_modes) 'm/mode' num2str(k) '.eps']);
    drawnow;
    %     eval( ['print -depsc ' num2str(nb_modes) 'm/' ...
    %         num2str(nb_modes_used) '_modes_used_for_estim_coef_correctif/mode' num2str(k) '.eps']);
    %     drawnow;
    
    
    
end

% keyboard;
% close all
