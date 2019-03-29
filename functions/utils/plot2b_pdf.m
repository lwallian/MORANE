function plot2b_pdf(i,j,param, bt_MCMC)
% plot a 2D pdf of chronos

%% Defining important parameters
m = param.nb_modes;
[N, ~, N_particules] = size(bt_MCMC);
dt = param.dt/1000;

%% Begin
bi = bt_MCMC(:,i,:);
bi = permute(bi, [1 3 2]); % N x N_particules
bj = bt_MCMC(:,j,:);
bj = permute(bj, [1 3 2]); % N x N_particules

%% Loop on time
for t=0:N-1
    bit = bi(t+1,:)'; % N_particules x 1
    bjt = bj(t+1,:)'; % N_particules x 1
    bt = [bit bjt]; % N_particules x 2
    ksdensity(bt);
    title(['joint density at t = ' num2str(t*dt)])
    pause(param.dt)
    
end

end