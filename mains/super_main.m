function super_main()
% Launch a set of simulations with a several set of parameters
% Especially several number of modes
%

init;
igrida=false;

% % % % figure;
% % % nb_modes_min=2;
% % % nb_modes_max=6;
% % % % % nb_modes_min=28;
% % % % % nb_modes_max=32;
% % % % vect_nb_modes = [8 16];
% % % % vect_nb_modes = nb_modes_min:2:nb_modes_max;
% % vect_nb_modes = 2.^(1:4);
% vect_nb_modes = 2.^(5);
vect_nb_modes = 2;
nb_modes_max = max(vect_nb_modes);


%% Type of data
% These 3D data give good results
% They are saved in only one file
% (~ 250 time step)
% % type_data = 'LES_3D_tot_sub_sample_blurred';
% type_data = 'incompact3d_wake_episode3_cut_truncated';
% % type_data = 'inc3D_Re3900_blocks_truncated';
% % type_data = 'inc3D_Re3900_blocks';
% type_data = 'incompact3D_noisy2D_40dt_subsampl';
type_data = 'incompact3D_noisy2D_40dt_subsampl_truncated';

% These 3D data are bigger, since the spatial grid is thinner
% and the number of time step is bigger
% (~ 2000 time step)
% They are saved in different files
% The internship should try to use these data
%     type_data = 'inc3D_Re3900_blocks';


coef_correctif_estim.learn_coef_a=false;

%% Without subsampling
% decor_by_subsampl.bool=false

a_t = [false];
% a_t = [true false];

%% With subsampling
save_all_bi=false;
decor_by_subsampl.bool=true;
%     cell_test_fct={'b'};
decor_by_subsampl.test_fct='b';
%     cell_test_fct={'b' 'db'};
%     v_threshold=[1 10]/1000;
%     v_threshold=1e-3;
%   v_threshold=[0.5 5]/1000;
if strcmp(type_data,'incompact3d_wake_episode3_cut_truncated') ...
        || strcmp(type_data,'incompact3d_wake_episode3_cut')
    v_threshold=0.0005
    modal_dt=false;
else
    v_threshold=1e-4
    modal_dt=true;
end
%      v_threshold=[1e-4 1e-5]
% %     v_threshold=[0.1 0.5 1 5 10]/1000;
%for j =1:2
%         for i=1:length(cell_test_fct)
decor_by_subsampl.meth='bt_decor';
decor_by_subsampl.choice_n_subsample='auto_shanon'; % need 'bt_decor' or  'a_estim_decor'
for q=1:length(v_threshold)
    %             for k = v_nb_modes
    for k=vect_nb_modes
        %     for k= (2.^(1:8))
        %     for k=nb_mode_min:2:nb_mode_max
        k
        decor_by_subsampl.n_subsampl_decor=nan;
        decor_by_subsampl.spectrum_threshold=v_threshold(q);
        %                 decor_by_subsampl.test_fct=cell_test_fct{i}
        main_full_sto_local(type_data,k,igrida,coef_correctif_estim,...
            save_all_bi,decor_by_subsampl,a_t);
    end
end
%end
%end
 
super_main_full_sto_modal_dt_2_test_basis_2D(...
    vect_nb_modes,type_data,v_threshold,modal_dt)

