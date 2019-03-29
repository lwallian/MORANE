function plot_modal_nrj(bt_MCMC, param, ...
    I_deter, L_deter, C_deter, ...
    I_sto, L_sto, C_sto, Cov_noises, F1, F2)
% plots different energy flow

filtered = true;
period = 5;

N_filter = 2 * floor( period / param.dt);
% N_filter = 6 * floor( period / param.dt);
filter = ones( N_filter,1);
filter=filter/sum(filter);

%% Missing subgrid operators ?
if nargin < 10 || any(isnan(F1(:))) || any(isnan(F2(:)))
    [F1, F2] = coefficients_sto(param);
    save(param.name_file_2nd_result,'F1','F2','-append')
end

%% Loading needed parameters
bt_mean = mean(bt_MCMC,3);
[N, m] = size(bt_mean);

theta_theta = Cov_noises(1:m,1:m); % m x m
alpha_alpha = Cov_noises(m+1:end,m+1:end); % m^2 x m^2
alpha_alpha = reshape(alpha_alpha, [m m m m]); % m x m x m x m
alpha_theta = Cov_noises(m+1:end, 1:m); % m^2 x m
alpha_theta = reshape(alpha_theta, [m m m]); % m x m x m

%% Plots parameters
LineWidth = 1;
FontSize = 10;
FontSizeTtitle = 11;
width=2;
height=1.5;
% width=1;
% height=0.7;
figure1=figure(1);
close(figure1)
figure('Units','inches', ...
    'Position',[0 0 4*width m*height], ...
    'PaperPositionMode','auto');
figure2=figure(2);
close(figure2)
figure('Units','inches', ...
    'Position',[4*width 0 4*width m*height], ...
    'PaperPositionMode','auto');

%% Starting the subplots
t  = param.dt*(0:N-1);
t_filter = param.dt*(((N_filter-1)/2):(N-1-(N_filter-1)/2));
for i=1:m
    normalized_factor = param.lambda(i)/period;
    j = 5*(i-1) + 1; % trick to organize subplots
    %% first plot
    Ki = theta_theta(i,i) * ones([N 1]);
    for p=1:m
        Ki = Ki + 2*alpha_theta(p,i,i)*bt_mean(:,p);
        for q=1:m
            Ki = Ki + alpha_alpha(p,i,q,i)*mean(bt_MCMC(:,p,:).*bt_MCMC(:,q,:),3);
        end
    end
    Ki = Ki/normalized_factor;
    if filtered
        Ki_filtered = conv(Ki,filter,'valid');
    end
    figure(1)
    subplot(m,5,j+1)
    plot(t, Ki/2);hold on;
    plot(t_filter, Ki_filtered/2,'r')
    figure(2)
    subplot(m,5,j+1)
    plot(t_filter, Ki_filtered/2)
    hold on; plot(t_filter, mean(Ki_filtered/2)*ones(size(Ki_filtered)),'--');
    title(['\hspace{0.5cm} $\frac{T}{\lambda_{' num2str(i) '}} E( K_{' num2str(i) '}^{2})$/2'],...
        'interpreter','latex')
    %% second plot
    Aibi = I_deter(i)*bt_mean(:,i);
    %     Aibi = Aibi + I_deter(i)*bt_mean(:,i);
    for p=1:m
        Aibi = Aibi + L_deter(p,i)*mean(bt_MCMC(:,i,:).*bt_MCMC(:,p,:),3);
        for q=1:m
            Aibi = Aibi + C_deter(p,q,i)*mean(bt_MCMC(:,p,:).*bt_MCMC(:,q,:).*bt_MCMC(:,i,:),3);
        end
    end
    Aibi = Aibi/normalized_factor;
    if filtered
        Aibi_filtered = conv(Aibi,filter,'valid');
    end
    %     Aibi = Aibi + I_deter(i)*bt_mean(:,i);
    figure(1)
    subplot(m,5,j+2)
    plot(t, -Aibi);hold on; % minus sign to switch back to Memin's notation
    plot(t_filter, -Aibi_filtered,'r') % minus sign to switch back to Memin's notation
    figure(2)
    subplot(m,5,j+2)
    plot(t_filter, -Aibi_filtered) % minus sign to switch back to Memin's notation
    hold on; plot(t_filter, mean( -Aibi_filtered)*ones(size(Ki_filtered)),'--');
    title(['\hspace{0.5cm} $\frac{T}{\lambda_{' num2str(i) '}} E( A_{' num2str(i) '}b_{' num2str(i) '})$'],...
        'interpreter','latex')
    %% third plot
    Dibi_1 = zeros([N 1]);
    for p=1:m
        Dibi_1 = Dibi_1 + F1(p,i)*mean(bt_MCMC(:,i,:).*bt_MCMC(:,p,:),3);
    end
    Dibi_1=Dibi_1/normalized_factor;
    if filtered
        Dibi_1_filtered = conv(Dibi_1,filter,'valid');
    end
    figure(1)
    subplot(m,5,j+3)
    plot(t, -Dibi_1);hold on;% minus sign to switch back to Memin's notation
    plot(t_filter, -Dibi_1_filtered,'r');% minus sign to switch back to Memin's notation
    figure(2)
    subplot(m,5,j+3)
    plot(t_filter, -Dibi_1_filtered);% minus sign to switch back to Memin's notation
    hold on; plot(t_filter, mean(-Dibi_1_filtered)*ones(size(Ki_filtered)),'--');
    title(['\hspace{0.5cm} $\frac{T}{\lambda_{' num2str(i) '}} E( D^{(1)}_{' num2str(i) '}b_{' num2str(i) '})$'],...
        'interpreter','latex')
    %% fourth plot
    if param.adv_corrected
        Dibi_2 = zeros([N 1]);
        for p=1:m
            Dibi_2 = Dibi_2 + F2(p,i)*mean(bt_MCMC(:,i,:).*bt_MCMC(:,p,:),3);
        end
        Dibi_2=Dibi_2/normalized_factor;
        if filtered
            Dibi_2_filtered = conv(Dibi_2,filter,'valid');
        end
        figure(1)
        subplot(m,5,j+4)
        plot(t, -Dibi_2);hold on;% minus sign to switch back to Memin's notation
        plot(t_filter, -Dibi_2_filtered,'r');% minus sign to switch back to Memin's notation
        figure(2)
        subplot(m,5,j+4)
        plot(t_filter, -Dibi_2_filtered);% minus sign to switch back to Memin's notation
        hold on; plot(t_filter, mean(-Dibi_2_filtered)*ones(size(Ki_filtered)),'--');
        title(['\hspace{0.5cm} $\frac{T}{\lambda_{' num2str(i) '}} E( D^{(2)}_{' num2str(i) '}b_{' num2str(i) '})$'],...
            'interpreter','latex')
    else
        Dibi_2 = 0;
        Dibi_2_filtered = 0;
    end
    %% 0-th plot
    Dibi_1 = zeros([N 1]);
    for p=1:m
        Dibi_1 = Dibi_1 + F1(p,i)*mean(bt_MCMC(:,i,:).*bt_MCMC(:,p,:),3);
    end
    Dibi_1=Dibi_1/normalized_factor;
    if filtered
        Dibi_1_filtered = conv(Dibi_1,filter,'valid');
    end
    figure(1)
    subplot(m,5,j)
    plot(t, Ki/2 - Aibi - Dibi_1 - Dibi_2);hold on;% minus sign to switch back to Memin's notation
    total = -Ki_filtered/2 - Aibi_filtered ...
        - Dibi_1_filtered - Dibi_2_filtered;
    plot(t_filter,total ,'r');% minus sign to switch back to Memin's notation
    figure(2)
    subplot(m,5,j)
    plot(t_filter, total);% minus sign to switch back to Memin's notation
    hold on; plot(t_filter, mean(total)*ones(size(total)),'--');
    title(['\hspace{0.5cm} Total flux'],...
        'interpreter','latex')
end

end