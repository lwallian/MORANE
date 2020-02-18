function [cmap,cax] = ...
    plot_iso_2dvort(param,name_simu, modal_dt,reconstruction,...
    big_T,first_big_T,Q,v_index_time)
%

param.MX = param.MX(1:2);
param.d=2;
X0=[0 1];
height = 2;
taille_police=10;

param = fct_name_reconstruction_2dvort(...
    param,modal_dt,reconstruction,name_simu);

sizQ=size(Q);
n1 = sizQ(end);

%%
init_caxis = ((big_T == first_big_T) && strcmp(name_simu,'ref'));
% init_caxis = ((big_T == first_big_T) );
% warning('DEBUG')
%%

if init_caxis
    boundcmap = [0.35 0.65];
% %     boundcmap = [0.3 0.7];
%     boundcmap = [0.25 0.75];
% %     % %     boundcmap = [0.1 0.9];
% %     %     boundcmap = [0 1];
end
factor_satur = 1.3;
% factor_satur = 1.5;



param.name_file_omega_mode = [ param.folder_data_PIV ...
    '2dvort_mode_' param.type_data '_' num2str(param.nb_modes) '_modes_PIV.mat'];
load(param.name_file_omega_mode,'x_unique_PIV','y_unique_PIV');

Q=reshape(Q,[param.MX n1]);
if param.data_assimilation == 2 && (1/param.viscosity == 300)
    x_cylinder=0;
%     %     param.name_file_omega_mode = [ param.folder_data_PIV ...
%     %         '2dvort_mode_' param.type_data '_' num2str(param.nb_modes) '_modes_PIV.mat'];
%     param.name_file_omega_mode = [ param.folder_data_PIV ...
%         '2dvort_mode_' param.type_data '_' num2str(param.nb_modes) '_modes_PIV.mat'];
%     load(param.name_file_omega_mode,'x_unique_PIV','y_unique_PIV');
    X = x_unique_PIV; Y = y_unique_PIV;
    x_cut = x_unique_PIV(1);
    if strcmp(param.type_data,'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_truncated')
        width = 9.5;
        % width = 10;
    else
        width = 8;
        %     width = 4;
    end
else
    x_cylinder=5;
    if strcmp(param.type_data,'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_truncated')
        x_cut = x_cylinder/2;
        width = 9.5;
        % width = 10;
    else
        x_cut=0;
        width = 8;
        %     width = 4;
    end
    
    X=param.dX(1)*(0:(param.MX(1)-1));
    X = X + x_cut;
    x_unique_PIV = x_unique_PIV + x_cylinder;
    Y=param.dX(2)*(0:(param.MX(2)-1));
    Y=Y-mean(Y);
    % [X,Y]=ndgrid(X,Y);
end



% vol_cyl = fct_remove_cylindar(ones(sizQ(1:2)), x_cylinder-x_cut,param.dX);
% % vol_cyl(:,:,1)=1;
% % vol_cyl(:,:,end)=1;
% % slice_cyl=vol_cyl(:,:,floor(size(vol_cyl,3)/2));
% % figure;imagesc(slice_cyl');axis xy;axis equal;
Q = fct_remove_cylindar(Q, x_cylinder-x_cut,param.dX);
% Q = fct_remove_cylindar(Q, x_cylinder/2,param.dX);

M=prod(param.MX);
Q=reshape(Q,[M n1]);
% vol_cyl=reshape(vol_cyl,[M 1]);


if init_caxis
    MQ=max(abs(Q(:)));
    cax =  MQ * [-1 1];
    %     MQ=max(Q(:));
    %     mQ=min(Q(:));
    %     deltaQ = MQ - mQ ;
    %     meanQ  = (MQ + mQ)/2 ;
    %     % v_threshold = 0;
    %     cax = meanQ + (deltaQ/2) * [-1 1];
    param.plot.omega.cax = cax;
else
    cax = param.plot.omega.cax;
end


% if init_caxis
%     MQ=max(Q(:));
%     mQ=min(Q(:));
%     deltaQ = MQ - mQ ;
%     % v_threshold = 0;
%     cax = mQ + deltaQ * cax
% else
if init_caxis
    cmap = parula;
    lsmaptot=size(cmap,1);
    nb_sat = (1-(boundcmap(2)-boundcmap(1)))/2;
    nb_sat = floor(nb_sat * lsmaptot);
    %     nb_sat = 10;
    lsmap=lsmaptot-2*nb_sat;
    cmapmin = cmap(1,:);
    cmapmax = cmap(end,:);
    cmap1 = interp1(1:lsmaptot,cmap(:,1),...
        linspace(1,lsmaptot, lsmap),...
        'spline');
    cmap2 = interp1(1:lsmaptot,cmap(:,2),...
        linspace(1,lsmaptot, lsmap),...
        'spline');
    cmap3 = interp1(1:lsmaptot,cmap(:,3),...
        linspace(1,lsmaptot, lsmap),...
        'spline');
    cmap((1+nb_sat):(end-nb_sat),:) = ...
        [ cmap1' cmap2' cmap3' ];
    cmap(1:nb_sat,:) = repmat(cmapmin,[nb_sat 1]);
    cmap((end-nb_sat+1):end,:) = repmat(cmapmax,[nb_sat 1]);
    
    cmap((lsmaptot/2):(lsmaptot/2+1),:)=...
        1-cmap((lsmaptot/2):(lsmaptot/2+1),:);
    cmap((lsmaptot/2):(lsmaptot/2+1),:) = ...
        cmap((lsmaptot/2):(lsmaptot/2+1),:)/factor_satur;
    cmap((lsmaptot/2):(lsmaptot/2+1),:)=...
        1-cmap((lsmaptot/2):(lsmaptot/2+1),:);
    %     cmap((lsmaptot/2):(lsmaptot/2+1),:)=1;
else
    cmap = param.plot.omega.cmap;
end

angle = linspace(0,2*pi,100);
R=0.5;
r=linspace(0,R,100);
[r,angle]=ndgrid(r,angle);
r=r(:);angle=angle(:);
c= cos(angle);s=sin(angle);
x_cyl=r.*c + x_cylinder;y_cyl=r.*s;

Q_save=Q;


if param.data_assimilation    
%     switch param.type_data
%         case 'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_truncated'
%             %             %             X_mes = x_unique_PIV(1+([10]));
%             %             %             Y_mes = y_unique_PIV(1+([10]));
%             %             %             X_mes = x_unique_PIV(1+([10 13 16]));
%             %             %             Y_mes = y_unique_PIV(1+([10 13 16]));
%             %             if (~isstruct(param.param_obs)) && isnan(param.param_obs)
%             %                 X_mes =[];Y_mes=[];
%             %             else
%             X_mes = x_unique_PIV( 1 + param.param_obs.x0_index + ...
%                 param.param_obs.subsampling_PIV_grid_factor * ...
%                 (1:param.param_obs.nbPoints_x) );
%             Y_mes = y_unique_PIV( 1 + param.param_obs.y0_index + ...
%                 param.param_obs.subsampling_PIV_grid_factor * ...
%                 (1:param.param_obs.nbPoints_y) );
            X_mes = x_unique_PIV( 1 + param.param_obs.x0_index + ...
                param.param_obs.subsampling_PIV_grid_factor * ...
                (0:(param.param_obs.nbPoints_x-1)) );
            Y_mes = y_unique_PIV( 1 + param.param_obs.y0_index + ...
                param.param_obs.subsampling_PIV_grid_factor * ...
                (0:(param.param_obs.nbPoints_y-1)) );
%             end
            
%         otherwise
%             X_mes = nan;
%             Y_mes = nan;
%     end
    [X_mes,Y_mes]=ndgrid(X_mes,Y_mes);
    X_mes=X_mes(:);
    Y_mes=Y_mes(:);
else
    X_mes=[];Y_mes=[];
end

% warning('only last snapshot of file plotted')
% for q=n1
for q=1:n1
    Q=Q_save(:,q);
    Q=reshape(Q,[param.MX 1]);
    if strcmp(param.type_data,...
            'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_truncated')
        Q(:,1,:)=[];
    end
    
    figure('Units','inches',...
        'Position',[X0(1) X0(2) width height],...
        'PaperPositionMode','auto');
    
    imagesc(X,Y,Q');
    
    caxis(cax);
    %     hold on;
    %     imagesc(X,Y,vol_cyl');
    hold on;
    plot(x_cyl,y_cyl,'.k');
    hold off;
    
    if strcmp(param.type_data,'DNS100_inc3d_2D_2018_11_16_blocks_truncated')
        axis([4 20 -2.5 2.5])
%         axis([2.5 20 -2.5 2.5])
    end
    
    set(gca,.....
        'Units','normalized',...
        'FontUnits','points',...
        'FontWeight','normal',...
        'FontSize',taille_police,...
        'FontName','Times')
%     set(gca,.....
%         'Units','normalized',...
%         'FontUnits','points',...
%         'FontWeight','normal',...
%         'FontSize',taille_police,...
%         'FontName','Times',...
%         'xtick',[],...
%         'xticklabel',[],...
%         'ytick',[],...
%         'yticklabel',[])
    colorbar;
    axis equal; axis xy;
    %%
    %     xlabel('x',...
    %         'FontUnits','points',...
    %         'interpreter','latex',...
    %         'FontSize',taille_police,...
    %         'FontName','Times')
    %     ylabel('y',...
    %         'FontUnits','points',...
    %         'interpreter','latex',...
    %         'FontSize',taille_police,...
    %         'FontName','Times')
    
    %     cmap = parula;
    %     cmap(1,:) = 0;
    %     cmap(end,:) = 1;
    %     colormap(cmap);
    colormap(cmap);
    colorbar;
    
    index_time = v_index_time(q);
%     index_time =  (big_T-first_big_T ) * ...
%         double(param.data_in_blocks.len_blocks ) + q;
    
    name_file =[ param.name_file_Reconstruction_omega ...
        num2str(index_time)];
%     name_file =[ param.name_file_Reconstruction_omega ...
%         '_big_T_' num2str(big_T) '_t_loc_' num2str(q)];

    time = param.dt * index_time;
    if isfield(param,'DA') && param.DA.bool
        bool_assimilation_step = ...
            any(index_time == param.DA.index_of_filtering);
    else
        bool_assimilation_step = false;
    end
    
    hold on;
    ax=axis;
    delta_ax = [ax(2)-ax(1) ax(4)-ax(3)];
    
    text(ax(1)+delta_ax(1)*0.1, ax(3)+delta_ax(2)*0.9,...
        [num2str(time) ' s']);
    if bool_assimilation_step
        t=text(ax(1)+delta_ax(1)*0.1, ax(3)+delta_ax(2)*0.1,...
            ['Obs.']);
        t(1).Color = 'red';
        t(1).FontSize = 14;
    end
    hold off
    
    %     param_from_file = param;
    %     if view_top
    %         name_file = [ name_file '_top'];
    %     else
    %         name_file = [ name_file '_side'];
    %     end
    %     %     if plot_arrow
    %     %         name_file = [ name_file '_arrow'];
    %     %     end
    %     if smooth
    %         name_file = [ name_file '_smooth'];
    %     end
    
    hold on;
    plot(X_mes,Y_mes,'.r','MarkerSize',10)
    %         plot(X_mes,Y_mes,'.r')
    hold off;
    
    drawnow
    pause(0.2)
    eval( ['print -loose -djpeg ' name_file '.jpg']);
    
    close all
%     warning('Returned forced');
%     return;
end

