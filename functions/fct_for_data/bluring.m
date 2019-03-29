%% Create low resolution data with high resolutions data

clear;
close all;
dbstop if error;

subsampling_factor = 3;
% bool_bluring=true;
bool_bluring=false;

load([ pwd '/data/data_incompact3d_wake_noisy_1_10dt.mat']);
U(:,:,2001:end,:)=[];
U(end,:,:,:)=[];
% U(:,end,:,:)=[];
% load([ pwd '/data/data_incompact3d_wake_noisy_1_sub_sampl.mat']);
% clear P Vort
% dX(3)=[];

% plot_1curl;
% pause;


[Mx,My,N,d]=size(U);
MX=[Mx My];

dX= subsampling_factor*dX;
len=floor(subsampling_factor/2);

if bool_bluring
    
    gauss=[ 1/4 1/2 1/4]; 
    
    i=1;
    Unew=nan([ Mx/subsampling_factor My N d] );
    whos;
    for k=1:Mx
        if mod(k,subsampling_factor)==0
            U_temp = U(max(1,k-len):min(end,k+len),:,:,:);
            if k-len >= 1 && k+len <= Mx
                U_temp = bsxfun( @times, U_temp, gauss');
                U_temp = squeeze(sum(U_temp,1));
            else
                U_temp = squeeze(mean(U_temp,1));
            end
            Unew(i,:,:,:)=U_temp;
            i=i+1;
        end
    end
    Mx=Mx/subsampling_factor;
    U=Unew; clear Unew i
    
    j=1;
    Unew=nan([ Mx My/subsampling_factor N d] );
    for q=1:My
        if mod(q,subsampling_factor)==0
            U_temp = U(:,max(1,q-len):min(end,q+len),:,:);
            if q-len >= 1 && q+len <= My
                U_temp = bsxfun( @times, U_temp, gauss);
                U_temp = squeeze(sum(U_temp,2));
            else
                U_temp = squeeze(mean(U_temp,2));
            end
            Unew(:,j,:,:)=U_temp;
            j=j+1;
        end
    end
    My=My/subsampling_factor;
    U=Unew; clear Unew U_temp j
    
else
    
    i=1;
    Unew=nan([ Mx/subsampling_factor My N d] );
    for k=1:Mx
        if mod(k,subsampling_factor)==0
            Unew(i,:,:,:)=U(k,:,:,:);
            i=i+1;
        end
    end
    Mx=Mx/subsampling_factor;
    U=Unew; clear Unew U_temp i
    
    j=1;
    Unew=nan([ Mx My/subsampling_factor N d] );
    for q=1:My
        if mod(q,subsampling_factor)==0
            Unew(:,j,:,:)=U(:,q,:,:);
            j=j+1;
        end
    end
    My=My/subsampling_factor;
    U=Unew; clear Unew j
end
% U=Unew; clear Unew i j
MX = MX/subsampling_factor;

save([ pwd '/data/data_incompact3d_wake_noisy_1_10dt_sub_sampl'], ...
    'U','dt','Re','dX','normalized');
% save([ pwd '/data/data_incompact3d_wake_noisy_1_sub_sampl_flou'], ...
%     'U','dt','Re','dX','normalized');


% plot_1curl;
