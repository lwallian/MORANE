function [v_threshold] = ...
    plot_isoQ(param,name_simu, modal_dt,reconstruction,...
    big_T,first_big_T,Q,v_index_time)
%

% zoom = 2;
zoom = 3;
% zoom=false;
% zoom=true;

v_threshold = [ 0  1.4907 0.3 ]
% v_threshold = [ 0  1.4907 0.2 ]
% % v_threshold = [ 0  1.4907 0.5 ]
% % % v_threshold = [ 0    1.4907 ]

% % big_T=81
% % warning('big_T=81 for now')
% init_threshold = ((big_T == first_big_T) && strcmp(name_simu,'ref'));
% if init_threshold
% % % %     v_threshold = [0.2 0.32 0.4];
% % % %     v_threshold = [0.25 0.32 0.4];
% % %     v_threshold = [0.27 0.32 0.4];
% %     v_threshold = [0.32 0.4];
%     v_threshold = [0.32 0.3];
% end

param = fct_name_reconstruction_Q(...
    param,modal_dt,reconstruction,name_simu);
% name_file_temp =[ param.name_file_Reconstruction_Q ...
%     num2str(big_T) '.mat'];
% load(name_file_temp,'Q')

n1 = length(v_index_time);
% n1 = size(Q);n1=n1(end);
d=param.d;

% % MQ=max(Q(:));
% % mQ=min(Q(:));
% % deltaQ = MQ - mQ ;
% % % v_threshold = 0;
% % v_threshold = mQ + deltaQ * [0.2 0.9];
% v_threshold = [0.3 0.7];

% %%
%
% slice_Q = Q(:,:,floor(size(Q,3)/2),1);
% x=param.dX(1)*(0:(param.MX(1)-1));
% y=param.dX(2)*(0:(param.MX(2)-1));y=y-mean(y);
% figure;imagesc(x,y,slice_Q');axis xy;axis equal;
% % keyboard;
%
% %%

% dbstop if error
plot_arrow = false;

taille_police = 12;
view_top = false;

if nargin == 0
    %     d=3;
    %     nb_mode=4;
    d=3;
    %     nb_mode=2;
end

smooth=false;

clear width2
% height2=1.1;
ratio_width = 1.5;
% ratio_width = 2;
switch zoom
    case 3
        height2=1.25;
        add_height2=1.3;
    case 2
        height2=1.25;
        add_height2=1.3;
    case 1
        height2=2;
        add_height2=1.5;
        %     height2=3;
        %     add_height2=1.5;
        %     height2=1;
        %     add_height2=0.4;
    case 0
        height2=1.5;
        add_height2=1.1;
        % add_height2=.8;
end
X0=[0 0];

dX=param.dX;
angle = (-0.1:0.1:2*pi);
c=cos(angle);
s=sin(angle);

% if d==3
Q=reshape(Q,[param.MX n1]);
x_cylinder=5;
sizQ=size(Q);

%     div_z=reshape(div_z,[M n1 2]);
% Cylinder
% % Rcyl= 0.5;
% % rcyl=Rcyl;
% % %     rcyl=(0:1/30:1)*Rcyl;
% x_cylinder=5;
% center = [x_cylinder 0];
% % %     x_cyl=5+0.5*c;
% % %     y_cyl=0.5*s;
% x_cyl=center(1)+bsxfun(@times,rcyl,c');
% x_cyl=x_cyl(:);
% y_cyl=center(2)+bsxfun(@times,rcyl,s');
% y_cyl=y_cyl(:);

x_cut = x_cylinder/2;
vol_cyl = fct_remove_cylindar(ones(sizQ(1:3)), x_cylinder-x_cut,param.dX);
vol_cyl(:,:,1)=1;
vol_cyl(:,:,end)=1;
% slice_cyl=vol_cyl(:,:,floor(size(vol_cyl,3)/2));
% figure;imagesc(slice_cyl');axis xy;axis equal;
Q = fct_remove_cylindar(Q, x_cylinder-x_cut,param.dX);
% Q = fct_remove_cylindar(Q, x_cylinder/2,param.dX);

M=prod(param.MX);
Q=reshape(Q,[M n1]);
vol_cyl=reshape(vol_cyl,[M 1]);

X=dX(1)*(ceil(x_cut/dX(1))+(1:(param.MX(1))));
%     X=dX(1)*(ceil(x_cut/dX(1))+(0:(param.MX(1)-1)));
Y=dX(2)*(-floor((param.MX(2)-1)/2):floor((param.MX(2)-1)/2));
Z=dX(3)*(-((param.MX(3)-1)/2):((param.MX(3)-1)/2));
% coef=5*n1;

% if init_threshold
%     MQ=max(Q(:));
%     mQ=min(Q(:));
%     deltaQ = MQ - mQ ;
%     % v_threshold = 0;
%     v_threshold = mQ + deltaQ * v_threshold
% else
%     v_threshold = param.plot.Q.v_threshold;
% end

switch zoom
    case 3
        Y_max = 2;
        idx_x=(X<15)&(X>4.5);
        X=X(idx_x);
        idx_y=(abs(Y)<Y_max);
        Y=Y(idx_y);
        idx= bsxfun(@and,idx_x' , idx_y);
        idx = repmat(idx, [ 1 1 param.MX(3)]);
        idx=idx(:);
        %         div_z = div_z(idx,:,:);
        Q = Q(idx,:);
        vol_cyl = vol_cyl(idx,:);
        %         V = V(idx,:,:,:);
        %         V=V(idx,:,:,:);
        %         D=D(idx,:,:);
        param.MX=[sum(idx_x) sum(idx_y) param.MX(3)];
        %         param.MX=[sum(idx_x) sum(idx_y)];
        M=prod(param.MX);
        axis_set = [ 4.5 15 -Y_max Y_max Z(1) Z(end)];
        %         axis_set = [ 4.5 10 -1.5 1.5];
        %         axis_set = [-1.5 1.5 4.5 10];
        
        nb_subsample_begin=10;
    case 2
        idx_x=(X<15)&(X>4.5);
        X=X(idx_x);
        idx_y=(abs(Y)<1.5);
        Y=Y(idx_y);
        idx= bsxfun(@and,idx_x' , idx_y);
        idx = repmat(idx, [ 1 1 param.MX(3)]);
        idx=idx(:);
        %         div_z = div_z(idx,:,:);
        Q = Q(idx,:);
        vol_cyl = vol_cyl(idx,:);
        %         V = V(idx,:,:,:);
        %         V=V(idx,:,:,:);
        %         D=D(idx,:,:);
        param.MX=[sum(idx_x) sum(idx_y) param.MX(3)];
        %         param.MX=[sum(idx_x) sum(idx_y)];
        M=prod(param.MX);
        axis_set = [ 4.5 15 -1.5 1.5 Z(1) Z(end)];
        %         axis_set = [ 4.5 10 -1.5 1.5];
        %         axis_set = [-1.5 1.5 4.5 10];
        
        nb_subsample_begin=10;
    case 1
        idx_x=(X<15)&(X>4.5);
        X=X(idx_x);
        idx_y=(abs(Y)<1.5);
        Y=Y(idx_y);
        idx= bsxfun(@and,idx_x' , idx_y);
        idx = repmat(idx, [ 1 1 param.MX(3)]);
        idx=idx(:);
        %         div_z = div_z(idx,:,:);
        Q = Q(idx,:);
        vol_cyl = vol_cyl(idx,:);
        %         V = V(idx,:,:,:);
        %         V=V(idx,:,:,:);
        %         D=D(idx,:,:);
        param.MX=[sum(idx_x) sum(idx_y) param.MX(3)];
        %         param.MX=[sum(idx_x) sum(idx_y)];
        M=prod(param.MX);
        axis_set = [ 4.5 10 -1.5 1.5 Z(1) Z(end)];
        %         axis_set = [ 4.5 10 -1.5 1.5];
        %         axis_set = [-1.5 1.5 4.5 10];
        
        nb_subsample_begin=10;
    case 0
        
        idx_x=(X<20)&(X>4.5);
        X=X(idx_x);
        idx_y=(abs(Y)<2.5);
        Y=Y(idx_y);
        idx= bsxfun(@and,idx_x' , idx_y);
        idx = repmat(idx, [ 1 1 param.MX(3)]);
        idx=idx(:);
        %         div_z = div_z(idx,:,:);
        Q = Q(idx,:);
        vol_cyl = vol_cyl(idx,:);
        %         V = V(idx,:,:,:);
        %         V=V(idx,:,:,:);
        %         D=D(idx,:,:);
        param.MX=[sum(idx_x) sum(idx_y) param.MX(3)];
        %         param.MX=[sum(idx_x) sum(idx_y)];
        M=prod(param.MX);
        nb_subsample_begin=10;
        axis_set = [ 4.5 20 -2.5 2.5 Z(1) Z(end)];
end
width2 = height2*(axis_set(2)-axis_set(1))/(axis_set(4)-axis_set(3));
if view_top
    height2=height2*1.1;
end

Q_save = Q;
X_save = X;
Y_save = Y;
Z_save = Z;
% V_save = V;
vol_cyl=reshape(vol_cyl,param.MX );

switch zoom
    case 1
        param.name_file_Reconstruction_Q = ...
            [ param.name_file_Reconstruction_Q 'zoom/'];
        mkdir(param.name_file_Reconstruction_Q);
    case 2
        param.name_file_Reconstruction_Q = ...
            [ param.name_file_Reconstruction_Q 'zoom2/'];
        mkdir(param.name_file_Reconstruction_Q);
    case 3
%         param.name_file_Reconstruction_Q = ...
%             [ param.name_file_Reconstruction_Q 'zoom3/'];
        mkdir(param.name_file_Reconstruction_Q);
end

% for q=1:1
for q=1:n1
    Q=Q_save(:,q);
    Q=reshape(Q,[param.MX 1]);
    
    [X,Y,Z]=ndgrid(X_save,Y_save,Z_save);
    
    
    height2=height2+add_height2;
    figure('Units','inches',...
        'Position',[X0(1) X0(2) ratio_width*width2 ratio_width*height2],...
        'PaperPositionMode','auto');
    height2=height2-add_height2;
    
    if smooth
        Q(:,:,:,1)=smooth3(Q(:,:,:,1));
    end
% %     %         %     threshold=6;
% %     %         %     [faces,verts,colors] = isosurface(X,Y,Z,Q,v_threshold(1));
% %     %         [faces,verts,colors] = isosurface(X,Y,Z,Q,threshold,Z);
% %     [faces,verts,colors] = isosurface(X,Y,Z,Q,0,Z);
%         [faces,verts,colors] = isosurface(X,Y,Z,Q,v_threshold(1),Z);
% %     [faces,verts,colors] = isosurface(X,Y,Z,Q,v_threshold(2),Z);
%     pp=patch('Vertices', verts, 'Faces', faces, ...
%         'FaceVertexCData', colors, ...
%         'FaceColor','interp', ...
%         'edgecolor', 'interp');
% %     set(pp,'FaceColor','green','EdgeColor','none','FaceAlpha',.8)
%     set(pp,'FaceColor','green','EdgeColor','none')
%     hold on;
    %     end
    
%     %     threshold=3;
    [faces,verts,colors] = isosurface(X,Y,Z,Q,v_threshold(2),Z);
%     [faces,verts,colors] = isosurface(X,Y,Z,Q,v_threshold(3),Z);
    %     [faces,verts,colors] = isosurface(X,Y,Z,Q,threshold,Z);
    pp=patch('Vertices', verts, 'Faces', faces, ...
        'FaceVertexCData', colors, ...
        'FaceColor','interp', ...
        'edgecolor', 'interp');
    set(pp,'FaceColor','red','EdgeColor','none');
    hold on;
    
    [faces,verts,colors] = isosurface(X,Y,Z,Q,v_threshold(3),Z);
    pp=patch('Vertices', verts, 'Faces', faces, ...
        'FaceVertexCData', colors, ...
        'FaceColor','interp', ...
        'edgecolor', 'interp');
    set(pp,'FaceColor','yellow','EdgeColor','none');
    
%     %     threshold=3;
%     [faces,verts,colors] = isosurface(X,Y,Z,Q,0,Z);
%     %     [faces,verts,colors] = isosurface(X,Y,Z,Q,threshold,Z);
%     pp=patch('Vertices', verts, 'Faces', faces, ...
%         'FaceVertexCData', colors, ...
%         'FaceColor','interp', ...
%         'edgecolor', 'interp');
% %     set(pp,'FaceColor','yellow','EdgeColor','none','FaceAlpha',.8)
%     set(pp,'FaceColor','yellow','EdgeColor','none')

%     %         %     threshold=6;
%     %         %     [faces,verts,colors] = isosurface(X,Y,Z,Q,v_threshold(1));
%     %         [faces,verts,colors] = isosurface(X,Y,Z,Q,threshold,Z);
%     [faces,verts,colors] = isosurface(X,Y,Z,Q,0,Z);
        [faces,verts,colors] = isosurface(X,Y,Z,Q,v_threshold(1),Z);
%     [faces,verts,colors] = isosurface(X,Y,Z,Q,v_threshold(2),Z);
    pp=patch('Vertices', verts, 'Faces', faces, ...
        'FaceVertexCData', colors, ...
        'FaceColor','interp', ...
        'edgecolor', 'interp');
%     set(pp,'FaceColor','green','EdgeColor','none','FaceAlpha',.8)
    set(pp,'FaceColor','green','EdgeColor','none')
    
    [faces,verts,colors] = isosurface(X,Y,Z,vol_cyl,0.5,Z);
    pp=patch('Vertices', verts, 'Faces', faces, ...
        'FaceVertexCData', colors, ...
        'FaceColor','interp', ...
        'edgecolor', 'interp');
    set(pp,'FaceColor',0.5*[1 1 1],'EdgeColor','none');
    
    
    % daspect([1,1,1])
    % % view(3); axis tight
    
    %     X=permute(X,[2 1 3]);
    %     Y=permute(Y,[2 1 3]);
    if ~view_top
        view(150,40);
        % view(160,50);
        % % view(30,50);
    end
    
    axis vis3d;
    axis equal
    % colormap winter
    camlight
    lighting phong
    
    hold on;
    ratio_com = ceil(3/height2);
    %     % for k=1:param.MX(3)
    %     for k=1:ratio_com:param.MX(3)
    %         plot3(x_cyl,y_cyl,Z_save(k)*ones(length(x_cyl)),'k','Linewidth',1)
    %     end
    axis equal
    hold off;
    
    %     set(gca,.....
    %         'Units','normalized',...
    %         'FontUnits','points',...
    %         'FontWeight','normal',...
    %         'FontSize',taille_police,...
    %         'FontName','Times')
    set(gca,.....
        'Units','normalized',...
        'FontUnits','points',...
        'FontWeight','normal',...
        'FontSize',taille_police,...
        'FontName','Times',...
        'xtick',[],...
        'xticklabel',[],...
        'ytick',[],...
        'yticklabel',[],...
        'ztick',[],...
        'zticklabel',[])
    
    if ~view_top
        axis(axis_set)
    end
    axis equal
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
    %%
    if ~view_top
        axis(axis_set)
    end
    
    
    index_time = v_index_time(q);
    
%     name_file =[ param.name_file_Reconstruction_Q ...

%         '_big_T_' num2str(big_T) '_t_loc_' num2str(q)];
    param_from_file = param;
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
    
%     index_time =  (big_T-first_big_T ) * ...
%         double(param.data_in_blocks.len_blocks ) + q
%     index_time = (big_T-1)*param.data_in_blocks.len_blocks + q;
    time = param.dt * index_time;
    if param.DA.bool
        bool_assimilation_step = any(index_time == param.DA.index_of_filtering);
    else
        bool_assimilation_step = false;
    end
    
    hold on;
    ax=axis;
    delta_ax = [ax(2)-ax(1) ax(4)-ax(3) ax(6)-ax(5)];
    
%     text(ax(1)+delta_ax(1)*0.1,ax(3)+delta_ax(2)*0.1, ax(5)+delta_ax(3)*0.1,...
%         [num2str(time) ' sec']);
    t1=text(ax(1)+delta_ax(1)*0.05,ax(3)+delta_ax(2)*0.05, ax(5)+delta_ax(3)*0.95,...
        [num2str(time) ' s']);
%     t1(1).FontSize
    t1(1).FontSize = 7;
%     text(ax(1)+delta_ax(1)*0.1,ax(3)+delta_ax(2)*0.1, ax(5)+delta_ax(3)*0.1,...
%         [num2str(time) ' sec after learning period']);
    if bool_assimilation_step
        t2=text(ax(1)+delta_ax(1)*0.1,ax(3)+delta_ax(2)*0.80, ax(5)+delta_ax(3)*0.1,...
            ['Obs.']);
        t2(1).Color = 'red';
        t2(1).FontSize = 10;
    end
    hold off
    
    name_file =[ param.name_file_Reconstruction_Q ...
        num2str(index_time)];
    
    drawnow
    eval( ['print -loose -djpeg ' name_file '.jpg']);
    %     eval( ['print -loose -depsc ' name_file '.eps']);
    
%     if q > 1
%         warning('Returned forced');
%         close all
%         return;
%     end
end
% % end
% if nargin ==0
%     keyboard;
% end
% % keyboard;
close all
end
