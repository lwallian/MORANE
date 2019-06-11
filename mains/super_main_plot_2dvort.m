function super_main_plot_2dvort(...
    vect_nb_modes,type_data,v_threshold,modal_dt,...
    no_subampl_in_forecast,vect_reconstruction,vect_adv_corrected)
% Launch a set of simulations with a several set of parameters
% Especially several number of modes
%

% super_main_from_existing_ROM_Simulation()

close all

if nargin == 0
    init;
    % % figure;
    % nb_modes_min=2;
    % nb_modes_max=32;
    % % nb_modes_max=2;
    
    % nb_modes_min=2
    % nb_modes_max=6
    % % % nb_modes_min=28
    % % % nb_modes_max=32
    % % vect_nb_modes = [8 16]
    % % vect_nb_modes = nb_modes_min:2:nb_modes_max
    % vect_nb_modes = 2.^5
    %     vect_nb_modes = 2.^(1:3)
    %             vect_nb_modes = 2.^(1:5)
    %     vect_nb_modes = 2;
    %     vect_nb_modes = [8];
    vect_nb_modes = 6
    %     vect_nb_modes = [16 8 6 4 2];
    %     vect_nb_modes = [2 4 6 8];
    %             vect_nb_modes = 2.^(1:4)
    %         vect_nb_modes = 2.^(1:4)
    %             vect_nb_modes = 2.^(1:6)
    
    % Type of data
    
    % % % These 3D data give good results
    % % % They are saved in only one file
    % % % (~ 250 time step)
    % % % type_data = 'LES_3D_tot_sub_sample_blurred';
    % % % type_data = 'incompact3d_wake_episode3_cut';
    % % %     type_data = 'incompact3D_noisy2D_40dt_subsampl';
    % % type_data = 'inc3D_Re3900_blocks';
    % type_data = 'incompact3d_wake_episode3_cut';
    %             type_data = 'incompact3d_wake_episode3_cut_truncated';
    % type_data = 'inc3D_Re3900_blocks_truncated';
    %     type_data = 'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_truncated'
    type_data = 'DNS100_inc3d_2D_2018_11_16_blocks_truncated'
    %         type_data = 'turb2D_blocks_truncated'
    
    no_subampl_in_forecast = false;
    % reconstruction = false;
    vect_reconstruction = [ false ]
    % vect_reconstruction = [true ]
    % adv_corrected = true
    vect_adv_corrected = [ false]
    %     vect_adv_corrected = [true false]
    %                 vect_adv_corrected = [true ]
    
    vect_DA = [ true ]
    vect_coef_bruit_obs = [0.1 0.8]
    %% With correctif coefficient
    
    % v_threshold=[1 10]/1000;
    switch type_data
        case {'DNS100_inc3d_2D_2018_11_16_blocks_truncated'}
            % Threshold used in the estimation of the optimal subsampling time step
            v_threshold=1e-6 % BEST
                        vect_modal_dt=1
%             vect_modal_dt=0:1
            
            %             %             v_threshold=[1e-4]
            %             %             v_threshold=[1e-6]
            %             %         v_threshold=[1e-6 1e-5 1e-4]
            %             %                     v_threshold=[1e-6 1e-5 1e-4]
            %             % if true, (mimic the use of a) disctinct subsampling time step for the
            %             % differentials equations of distincts chronos
            %             %             modal_dt=true
            %             vect_modal_dt=0:1
            % %             vect_modal_dt=1
            % %             vect_modal_dt=false
        case {'incompact3d_wake_episode3_cut',...
                'incompact3d_wake_episode3_cut_truncated'}
            v_threshold=0.0005
            %         v_threshold=1e-4
            %         v_threshold=1e-6
            v_threshold=[1e-6]
            warning('threshold changed')
            vect_modal_dt = false
        case {'inc3D_Re3900_blocks',...
                'inc3D_Re3900_blocks_truncated'}
            v_threshold=1e-4
            vect_modal_dt = true
        case 'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_truncated'
            % Threshold used in the estimation of the optimal subsampling time step
            v_threshold=1e-4 % BEST
            %             vect_modal_dt=true % BEST
            vect_modal_dt=0:1
            
            %             v_threshold=[1e-1 1e-2]
            %             vect_modal_dt=0:2
            
            %             v_threshold=[ 5e-4 ]
            %             v_threshold=[ 1e-5 5e-4 ]
            %             v_threshold=[5e-4 1e-4 1e-5]
            %                         v_threshold=1e-4
            %             v_threshold=[1e-4 5e-4 1e-3] %
            %                         v_threshold=[1e-3 5e-4 1e-4 1e-5]
            %             v_threshold=[ 1e-5  ]
            %             v_threshold=[1e-4] % best for now
            %             v_threshold=[1e-3 1e-5 ] % best for now
            %                         v_threshold=[1e-4 1e-3]
            %                         v_threshold=[1e-4 5e-4 1e-3]
            %                         v_threshold=[1e-5 1e-4 5e-4 1e-3 1e-2]
            % %         v_threshold=0.0005
            %             v_threshold=5e-4 % best for now
            %         v_threshold=1e-4 % comme 5e-4, sauf pour n=2 ,
            %         % % |=> pas de random => la dist min n'est pas plus peitet que le biais
            %         v_threshold=1e-5% diverge
            %         v_threshold=1e-6 % diverge encore +
            %             warning('threshold changed')
            % if true, (mimic the use of a) disctinct subsampling time step for the
            % differentials equations of distincts chronos
            %             vect_modal_dt=0:2
            %                                     modal_dt=true
            %             modal_dt=false
        case 'turb2D_blocks_truncated'
            v_threshold= [1e-5]
            %             v_threshold= [1e-3 1e-4 1e-5];
            vect_modal_dt=0:1
            %             vect_modal_dt=false
        otherwise
            % Threshold used in the estimation of the optimal subsampling time step
            v_threshold=0.0005
            % v_threshold=1e-6
            warning('default threshold')
            % if true, (mimic the use of a) disctinct subsampling time step for the
            % differentials equations of distincts chronos
            vect_modal_dt=false
    end
else
    vect_modal_dt = modal_dt;
end
nb_modes_max = max(vect_nb_modes);

%% Compute topos vorticity and rate of strain tensors
current_pwd = pwd; cd ..
param.folder_data = [ pwd '/data/' ];
cd(current_pwd); clear current_pwd
for k=vect_nb_modes
    name_file_tensor_mode = [ param.folder_data ...
        '2dvort_mode_' type_data '_' num2str(k) '_modes.mat'];
    if ~ (exist(name_file_tensor_mode,'file') == 2)
        % Compute topos gradients
        [param_temp,dphi_m_U] = grad_topos(type_data,k);
        % Compute topos vorticity and rate of strain tensors
        vort_topos(param_temp,dphi_m_U);
    end
end

%% Compute instateneous Q criterion
for coef_bruit_obs = vect_coef_bruit_obs
    for DA = vect_DA
        for modal_dt=vect_modal_dt
            for adv_corrected=vect_adv_corrected
                for reconstruction = vect_reconstruction
                    for q=1:length(v_threshold)
                        close all
                        pause(1)
                        for k=vect_nb_modes
                            %     for k=nb_modes_min:2:nb_modes_max
                            %         main_full_sto_modal_dt(k,v_threshold(q))
                            %                     main_from_existing_ROM_Simulation(type_data,k,...
                            main_plot_2dvort(type_data,k,...
                                v_threshold(q),...
                                no_subampl_in_forecast,reconstruction,...
                                adv_corrected,modal_dt,...
                                DA,coef_bruit_obs)
                            
                            %                     switch type_data
                            %                         case 'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_truncated'
                            %                             ax = axis;
                            %                             %                         ax(2)=40;
                            %                             ax(2)=20;
                            %                             axis(ax);
                            %                         case 'turb2D_blocks_truncated'
                            %                             ax = axis;
                            %                             ax(2)=3e7;
                            %                             axis(ax);
                            %                     end
                        end
                        
                        %% Save plot
                        %                 folder_results = [ pwd '/resultats/current_results/summary/'];
                        %                 current_pwd = pwd; cd ..
                        %                 folder_data = [ pwd '/data/' ];
                        %                 cd(current_pwd);
                        %                 % folder_results = ['/Users/Resseguier/Documents/MATLAB/POD/all/resultats/current_results/'];
                        %                 % folder_data = '/Users/Resseguier/Documents/MATLAB/POD/data/';
                        %                 % folder_results = ['/Users/vressegu/Documents/matlab/POD-NS_Stochastique/current_used/' ...
                        %                 %         'all/resultats/current_results/'];
                        %                 % %     param.folder_results =  [ pwd '/resultats/current_results/'];
                        %                 %     eval( ['print -depsc ' folder_results 'sum_modes_n=' num2str(nb_modes_max) ...
                        %                 %             '_threshold_' num2str(v_threshold(q)) '.eps']);
                        %
                        %                 threshold = num2str(v_threshold(q));
                        %                 iii = (threshold =='.');
                        %                 threshold(iii)='_';
                        %
                        %                 %             str = ['print -depsc ' folder_results type_data '_sum_modes_n=' ...
                        %                 str = ['print -dpng ' folder_results type_data '_sum_modes_n=' ...
                        %                     num2str(nb_modes_max) '_threshold_' threshold ...
                        %                     '_fullsto'];
                        %                 %             if modal_dt
                        %                 %                 str =[ str '_modal_dt'];
                        %                 %             end
                        %                 if modal_dt == 1
                        %                     str =[ str '_modal_dt'];
                        %                 elseif modal_dt == 2
                        %                     str =[ str '_real_dt'];
                        %                 end
                        %                 if ~ adv_corrected
                        %                     str =[ str '_NoAdvCorect'];
                        %                 end
                        %                 if reconstruction
                        %                     str =[ str '_reconstruction'];
                        %                 else
                        %                     str =[ str '_forecast'];
                        %                 end
                        %                 str =[ str '.png'];
                        %                 %             str =[ str '.eps'];
                        %                 str
                        %                 drawnow
                        %                 pause(1)
                        %                 eval(str);
                        %
                        %                 %     eval( ['print -depsc ' folder_results type_data '_sum_modes_n=' ...
                        %                 %         num2str(nb_modes_max) '_threshold_' threshold ...
                        %                 %         '_fullsto_modal_dt.eps']);
                    end
                end
            end
        end
    end
end
