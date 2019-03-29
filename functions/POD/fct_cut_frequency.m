function n_subsampl_decor = fct_cut_frequency(bt,lambda,param)
% Compute how much we can subsample in time the resolved modes with respect to the Shanon
% criterion

if isfield(param,'N_estim')
    bt=bt(1:param.N_estim,:);
end

N_tot=size(bt,1);
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
% Keep, for each frequency, the maximum of all modes
spectrum=max(spectrum,[],2);

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
% if strcmp(param.decor_by_subsampl.test_fct,'b')
%     freq_cut = 2 * freq_cut; % Because the evolution equation of temporals modes are quadratics
%     % So, the maximal frequency of the derivates of temporal modes is the
%     % double of the maximal frequency of the temporal modes
%     % Thus, the minimal frequency one should use is the double of the maximal
%     % frequency of the temporal modes.
% end

% Shanon criterion allow the following subsample
n_subsampl_decor=1./(2*freq_cut); % Shanon criterion

% Keep this subsample rate in the interval of possible values
n_subsampl_decor=floor(n_subsampl_decor);
n_subsampl_decor=min(n_subsampl_decor,N_tot);
n_subsampl_decor=max(n_subsampl_decor,1);

%%
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
