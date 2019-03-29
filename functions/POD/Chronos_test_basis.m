function [param_ref, bt,truncated_error2]=Chronos_test_basis(param_ref)
% Compute the spatial modes phi of the POD, the corresponding temporal coefficients bt,
% the temporal mean m_U, the time subsampling and
% the residual velocity U neglected by the Galerkin projection
%

name_file = ...
    [param_ref.folder_data param_ref.type_data ...
    '_' num2str(param_ref.nb_modes) '_modes' ...
    '_threshold_' num2str(param_ref.decor_by_subsampl.spectrum_threshold) ...
    '_nb_period_test_' num2str(param_ref.nb_period_test) ...
    '_Chronos_test_basis.mat'];

if exist(name_file,'file')==2
    load(name_file);
else
    %% Load
    param=param_ref;
    param.name_file_mode=['mode_' param.type_data '_' ...
        num2str(param.nb_modes) '_modes.mat'];
    param.name_file_mode=[ param.folder_data param.name_file_mode ];
    load(param.name_file_mode)
    % Remove the time average value m_U
    m_U = phi_m_U(:,param.nb_modes+1,:);
    phi_m_U(:,param.nb_modes+1,:)=[];
    phi=phi_m_U; clear phi_m_U
    phi = permute(phi,[1 4 3 2]);
    
    n_subsampl = param.decor_by_subsampl.n_subsampl_decor;
    
    if isfield(param_ref,'data_in_blocks') && ...
            isfield(param_ref.data_in_blocks,'bool') && ...
            param_ref.data_in_blocks.bool
        % if data are saved in several files
        len_blocks=param.data_in_blocks.len_blocks;
%         param_temp = read_data_blocks(...
%             [ param.type_whole_data(1:(end-10)) '_test_basis' ],...
%             param.folder_data);
        param_temp = read_data_blocks(...
            [ param.type_data(1:(end-10)) '_test_basis' ],...
            param.folder_data);
        N_tot= param_temp.data_in_blocks.nb_blocks ...
            * param_temp.data_in_blocks.len_blocks;
        clear param_temp
        N_tot= floor(N_tot/n_subsampl);
    else
%         param.data_in_blocks.bool = false;
%         len_blocks=inf;
%         param.data_in_blocks.nb_blocks = [];
%         param.type_data = [ param.type_data(1:(end-10)) '_test_basis' ];
%         param.data_in_blocks.type_whole_data = param.type_data;
        param.data_in_blocks.bool = false;
        len_blocks=inf;
        param.data_in_blocks.nb_blocks = [];
        param.type_data = [ param.type_data(1:(end-10)) '_test_basis' ];
        param.data_in_blocks.type_whole_data = param.type_data;
        % Load new file
        U=read_data(param.type_data,param.folder_data, ...
            param.data_in_blocks.type_whole_data,param.modified_Re);
        U = bsxfun(@minus, U , m_U);
        siz = size(U);
        N_tot= floor(siz(end-1)/n_subsampl);
    end
    
    % Initialization
    t_local=1; % index of the snapshot in a file
%     t_local=0; % index of the snapshot in a file
    t_subsample=1;
    %     if isfield(param_ref,data_in_blocks.bool) && ...
    %             param_ref.data_in_blocks.bool
    %         big_T = param.data_in_blocks.nb_blocks; % index of the file
    %     else
    %         big_T = [];
    %     end
    truncated_error2 = nan([1 N_tot]);
    bt = nan([N_tot 1 1 param.nb_modes]);
    big_T = param.data_in_blocks.nb_blocks; % index of the file
    
    for t=1:n_subsampl*N_tot % loop for all time
        if param.data_in_blocks.bool && ...
                ((t_local == len_blocks + 1) || t==1) % A new file needs to be loaded
            % initialization of the index of the snapshot in the file
            t_local=1;
            % Incrementation of the file index
            big_T = big_T+1;
            % Name of the new file
            param.type_data=[param.data_in_blocks.type_whole_data num2str(big_T)];
            % Load new file
            U=read_data(param.type_data,param.folder_data, ...
                param.data_in_blocks.type_whole_data,param.modified_Re);
            
            U = bsxfun(@minus, U , m_U);
            
%             if t ==1 && ...
%                 ~ ( isfield(param_ref,'data_in_blocks') && ...
%                     isfield(param_ref.data_in_blocks,'bool') && ...
%                     param_ref.data_in_blocks.bool )
%                 siz = size(U);
%                 N_tot= siz(end-1);
%                 truncated_error2 = nan([1 N_tot]);
%                 bt = nan([N_tot 1 1 param.nb_modes]);
%             end
        end
        
        if mod(t-1,n_subsampl)==0
%         if mod(t,n_subsampl)==1
            bt(t_subsample,1,1,:) = sum(sum(bsxfun(@times, U(:,t_local,:), phi),1),3) ...
                * prod(param.dX);
            
            U(:,t_local,:) = U(:,t_local,:) ...
                - sum( bsxfun(@times, phi, bt(t_subsample,1,1,:)) ,4);
            
            truncated_error2(1,t_subsample) = sum(sum(U(:,t_local,:).^2,1),3) ...
                * prod(param.dX);
            
            % Incrementation of the subsampled time
            t_subsample=t_subsample+1;
        end
        
        % Incrementation of the index of the snapshot in the file
        t_local=t_local+1;
    end
    clear U phi
    bt = permute(bt,[1 4 2 3]);
    truncated_error2 = truncated_error2';
    save(name_file,'truncated_error2','bt');
end



param_ref.truncated_error2 = truncated_error2;

end
