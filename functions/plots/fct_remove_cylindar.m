function temp = fct_remove_cylindar(temp, x_cylinder,dX)

D=1;
[Mx,My,Mz,n1]=size(temp);
D_pixel = D./dX;
idx_x_center_cyl = x_cylinder*D_pixel(1);
idx_y_center_cyl = (My+1)/2;
idx_x_cyl = floor(idx_x_center_cyl-D_pixel(1)/2):ceil(idx_x_center_cyl+D_pixel(1)/2);
idx_y_cyl = floor(idx_y_center_cyl-D_pixel(2)/2):ceil(idx_y_center_cyl+D_pixel(2)/2);
for k=1:n1
    for i=idx_x_cyl
        for j=idx_y_cyl
            dist=sqrt( ((i-idx_x_center_cyl)*dX(1))^2 ...
                + ((j-idx_y_center_cyl)*dX(2))^2 );
            if dist<= D/2
%             dist=sqrt( (i-idx_x_center_cyl)^2 + (j-idx_y_center_cyl)^2 );
%             if dist<= D_pixel/2
                temp(i,j,:,k)=zeros(1,1,Mz,1);
            end
        end
    end
end