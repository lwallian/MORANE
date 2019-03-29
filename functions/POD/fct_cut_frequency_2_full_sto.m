function [rate_dt, ILC,pchol_cov_noises ] = ...
    fct_cut_frequency_2_full_sto(bt,ILC,param,pchol_cov_noises,modal_dt)
% Compute how much we can subsample in time the resolved modes with respect to the Shanon
% criterion

if nargin < 5
    modal_dt = true;
end

[N_tot,nb_modes]=size(bt);
lambda=param.lambda;
% Normalization
bt = bsxfun( @times, 1./sqrt(lambda)' , bt);
% If the criterion is on the derivative of the Chronos
if strcmp(param.decor_by_subsampl.test_fct,'db')
    % Derivative of Chronos
    bt =diff_bt(bt,param.dt);
end
spectrum=abs(fft(bt)).^2;
% Keep the first half of the spectrum (positive frequencies)
spectrum = spectrum(1:ceil(N_tot/2),:);
freq=(0:(ceil(N_tot/2)-1))/N_tot;
% % Keep, for each frequency, the maximum of all modes
% spectrum=max(spectrum,[],2);

% Initialization
spectrum_tot = spectrum; clear spectrum
rate_dt=nan(nb_modes,1);

for k=1:nb_modes % loop on number of modes
    
    spectrum = spectrum_tot(:,k);
    
    % Threshold to determine the numerical zero
    spectrum_threshold=param.decor_by_subsampl.spectrum_threshold;
    max_s=max(spectrum);
    threshold=max_s*spectrum_threshold;
    
    % Find from which frequency the maximum spectrum is null
    idx =( spectrum > threshold);
    idx_temp = find(idx);
    idx_temp = min(idx_temp(end)+1,length(freq));
    freq_cut = freq(idx_temp);
    % If the criterion is on the Chronos
    
%     if strcmp(param.decor_by_subsampl.test_fct,'b')
%         freq_cut = 2 * freq_cut; % Because the evolution equation of temporals modes are quadratics
% %         So, the maximal frequency of the derivates of temporal modes is the
% %         double of the maximal frequency of the temporal modes
% %         Thus, the minimal frequency one should use is the double of the maximal
% %         frequency of the temporal modes.
%     end
    
    % Shanon criterion allow the following subsample
    n_subsampl_decor=1./(2*freq_cut); % Shanon criterion
    
    % Keep this subsample rate in the interval of possible values
%     n_subsampl_decor=floor(n_subsampl_decor);
    n_subsampl_decor=min(n_subsampl_decor,N_tot);
    n_subsampl_decor=max(n_subsampl_decor,1);
    
    rate_dt(k)=n_subsampl_decor;
end

if modal_dt == 2
    rate_dt = min(rate_dt) * ones(size(rate_dt));
end

fprintf(['The time step is modulated by ' num2str(rate_dt') '\n']);

%% Modify Chronos evolution equation

% Finite variation terms

I_deter=ILC.deter.I;
L_deter=ILC.deter.L;
C_deter=ILC.deter.C;

I_sto=ILC.sto.I;
L_sto=ILC.sto.L;
C_sto=ILC.sto.C;

C_sto= bsxfun( @times,C_sto,permute(rate_dt,[3 2 1]));

L_sto= bsxfun( @times,L_sto,rate_dt(:,1)');

for q=1:nb_modes
    I_sto(q)=-trace(diag(lambda)*C_sto(:,:,q));
end
ILC.modal_dt.I=I_sto+I_deter;
ILC.modal_dt.L=L_sto+L_deter;
ILC.modal_dt.C=C_sto+C_deter;

% Martingale terms
r_rate_dt = sqrt(rate_dt);
weight = repmat(r_rate_dt',[nb_modes 1]);
weight = [ r_rate_dt weight ];
weight = weight(:);
pchol_cov_noises = bsxfun(@times, weight, pchol_cov_noises);
    
    %
    function db=diff_bt(b,dt)
    % Computation of the time derivative of Chronos
    %
    db=nan(size(b));
    db(1,:)=b(2,:)-b(1,:);
    db(2:end-1,:)=1/2*(b(3:end,:)-b(1:end-2,:));
    db(end,:)=b(end,:)-b(end-1,:);
    db=1/dt*db;
    end
    
end
