function super_main_from_existing_ROM_Simulation(...
    vect_nb_modes,type_data,v_threshold,vect_modal_dt,...
    no_subampl_in_forecast,vect_reconstruction,vect_adv_corrected,...
    test_fct,vect_svd_pchol,eq_proj_div_free)
% Launch a set of simulations with a several set of parameters
% Especially several number of modes
%

close all


if nargin == 0
    init;
    global choice_n_subsample;
    global stochastic_integration;
    global estim_rmv_fv;
    global correlated_model;
    
    %% Number of modes for the the ROM
%     vect_nb_modes = [2 4 6 8]
    vect_nb_modes = [16]
    % vect_nb_modes = 2.^(1:4)
    no_subampl_in_forecast = false;
    vect_reconstruction = [ false] % for the super_main_from_existing_ROM
%     vect_adv_corrected = [ true false]
    vect_adv_corrected = [ true]
    
    % To choose between the shannon and correlation time downsampling
    % methods
%     choice_n_subsample = 'auto_shannon';
    choice_n_subsample = 'auto_shanon'; % 'auto_shanon' 'htgen' 'lms'
    stochastic_integration = 'Str'; % 'Str'  'Ito'
    estim_rmv_fv = false;
    test_fct ='b';
    vect_svd_pchol = true
    correlated_model = false
    
    % Projection on the free-divergence-function space
    % 0 : no projection / 1 : projection of deterministic terms
    %  / 2 :projection of noise terms
    eq_proj_div_free = 1;
    
    %% Type of data
    % Other datasets (do not use)
    % % type_data = 'LES_3D_tot_sub_sample_blurred';
    % % type_data = 'incompact3d_wake_episode3_cut_truncated';
    % % type_data = 'inc3D_Re3900_blocks_truncated';
    % % type_data = 'inc3D_Re3900_blocks';
    % % type_data = 'incompact3D_noisy2D_40dt_subsampl';
    % % type_data = 'incompact3D_noisy2D_40dt_subsampl_truncated';
    % % type_data = 'inc3D_Re300_40dt_blocks_truncated';
    % % type_data = 'turb2D_blocks_truncated'
    % % type_data = 'test2D_blocks_truncated'
    % % type_data = 'DNS300_inc3d_3D_2017_04_02_blocks_truncated'
    % % type_data = 'inc3D_HRLESlong_Re3900_blocks_truncated'
    % % type_data = 'small_test_in_blocks_truncated'
    % % type_data = 'test_1_block'
    % % type_data = 'inc3D_Re3900_blocks';
    % % type_data = 'turb2D_blocks_truncated'
    
    % These 3D data ( Re 300) gives good results
    type_data = 'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_truncated'
    
    % These 2D data ( Re 100) gives good results
%     type_data = 'DNS100_inc3d_2D_2018_11_16_blocks_truncated'
    
    % Small dataset for debuging
    % type_data = 'incompact3D_noisy2D_40dt_subsampl_truncated';
    % % type_data = 'incompact3d_wake_episode3_cut_truncated';
    
    %% Important parameters
    switch type_data
        %  - Threshold used in the estimation of the optimal subsampling time step
        % - if modal-dt = true,
        %   (mimic the use of a) disctinct subsampling time step for the
        %   differentials equations of distincts chronos
        case {'incompact3D_noisy2D_40dt_subsampl_truncated'}
            v_threshold=[1e-5]
            vect_modal_dt=false
        case {'DNS100_inc3d_2D_2018_11_16_blocks_truncated'}
            % Threshold used in the estimation of the optimal subsampling time step
            v_threshold=1e-6 % BEST
            vect_modal_dt=1
        case 'turb2D_blocks_truncated'
            v_threshold= [1e-5]
            vect_modal_dt=0:1
        case {'incompact3d_wake_episode3_cut_truncated',...
                'incompact3d_wake_episode3_cut'}
            v_threshold=1e-6
            % %         v_threshold=1e-4
            vect_modal_dt=false
            %         vect_modal_dt=true;
        case {'LES_3D_tot_sub_sample_blurred',...
                'inc3D_Re3900_blocks',...
                'inc3D_Re3900_blocks_truncated'}
            v_threshold=1e-3
            vect_modal_dt=true
        case 'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_truncated'
            v_threshold=1e-4 % BEST
            vect_modal_dt=true;
        otherwise
            v_threshold=0.0005
            vect_modal_dt=false
    end
    
    if strcmp(choice_n_subsample, 'corr_time')
        v_threshold = NaN;
    end
else
    global choice_n_subsample;
    global stochastic_integration;
    global estim_rmv_fv;
    global correlated_model;
end
nb_modes_max = max(vect_nb_modes);

for svd_pchol=vect_svd_pchol
for modal_dt=vect_modal_dt
    for adv_corrected=vect_adv_corrected
        for reconstruction = vect_reconstruction
            for q=1:length(v_threshold)
                close all
                pause(1)
                for k=vect_nb_modes
                    main_from_existing_ROM_Simulation(type_data,k,...
                        v_threshold(q),...
                        no_subampl_in_forecast,reconstruction,...
                        adv_corrected,modal_dt,test_fct,svd_pchol,eq_proj_div_free)
                    
                    switch type_data
                        case 'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_truncated'
                            ax = axis;
                                                    ax(2)=40;
%                             ax(2)=20;
                            axis(ax);
                        case 'turb2D_blocks_truncated'
                            ax = axis;
                            ax(2)=3e7;
                            axis(ax);
                    end
                end
                
                fig1=figure(1);
                ax=axis;
                fig1.Position = [0 0 4*(ax(2)-ax(1))/20 4];
%                 gca(fig1,'Units','inches', ...
%                     'Position',[0 0 4*(ax(2)-ax(1))/20 4], ...
%                     'PaperPositionMode','auto');
                %% Save plot
                folder_results = [ pwd '/resultats/current_results/summary/'];
                current_pwd = pwd; cd ..
                folder_data = [ pwd '/data/' ];
                cd(current_pwd);
                
                threshold = num2str(v_threshold(q));
                iii = (threshold =='.');
                threshold(iii)='_';
                
                dir = [ folder_results type_data '/sum_modes_n=' ...
                    num2str(nb_modes_max) ];
                mkdir(dir);
                
                switch choice_n_subsample
                    case 'auto_shanon'
                        str = ['print -dpng ' dir '/' choice_n_subsample ...
                            'threshold_' threshold ...
                            '_fullsto'];
                    otherwise
                        str = ['print -dpng ' dir '/' choice_n_subsample ...
                            'auto_corr_time_fullsto'];
                end
%                 str = ['print -dpng ' folder_results type_data '_sum_modes_n=' ...
%                     num2str(nb_modes_max) 'auto_corr_time_fullsto'];

                if modal_dt == 1
                    str =[ str '_modal_dt'];
                elseif modal_dt == 2
                    str =[ str '_real_dt'];
                end
                if ~ adv_corrected
                    str =[ str '_NoAdvCorect'];
                end
                if reconstruction
                    str =[ str '_reconstruction'];
                else
                    str =[ str '_forecast'];
                end
                str = [str '_integ_' stochastic_integration];
                if estim_rmv_fv
                    str=[str '_estim_rmv_fv'];
                end
                if svd_pchol
                    str=[str '_svd_pchol'];
                end
                if eq_proj_div_free == 2
                    str=[str '_DFSPN'];                    
                end
                if correlated_model
                    str = [str '_correlated']
                end
                str =[ str '.png'];
                str
                drawnow
                pause(1)
                eval(str);
            end
        end
    end
end
end
end
