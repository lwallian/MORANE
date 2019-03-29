function super_main_finition()
% Launch a set of simulations with a several set of parameters
% Especially several number of modes
%

init;
%igrida=false;
igrida=true;
nb_mode_max=16;

coef_correctif_estim.learn_coef_a=false;

%% Without subsampling
% decor_by_subsampl.bool=false

a_t = [false];
% a_t = [true false];
save_all_bi=false;
% save_all_bi=true;
% for k=2:2:nb_mode_max
    
    %     decor_by_subsampl.bool=false
    %     for j =1:2
    %         main(k,igrida,coef_correctif_estim,save_all_bi,decor_by_subsampl,a_t(j))
    %         save_all_bi=false;
    %     end
    
    %% With subsampling
    save_all_bi=false;
    decor_by_subsampl.bool=true;
    %     cell_test_fct={'b'};
    decor_by_subsampl.test_fct='b';
    %     cell_test_fct={'b' 'db'};
%     %     v_threshold=[1 10]/1000;
%     %     v_threshold=1e-3;
%     %   v_threshold=[0.5 5]/1000;
% %     v_threshold=[0.1 0.5 1 5 10]/1000;
%      v_threshold=[1e-4 1e-5]
     v_threshold=[1e-5]
    %for j =1:2
        %         for i=1:length(cell_test_fct)
        decor_by_subsampl.meth='bt_decor';
        decor_by_subsampl.choice_n_subsample='auto_shanon'; % need 'bt_decor' or  'a_estim_decor'
        for q=1:length(v_threshold)
%             for k=2:2:nb_mode_max
            for k=16:2:nb_mode_max
                decor_by_subsampl.n_subsampl_decor=nan;
                decor_by_subsampl.spectrum_threshold=v_threshold(q);
%                 decor_by_subsampl.test_fct=cell_test_fct{i}
                main_full_sto_local(k,igrida,coef_correctif_estim,save_all_bi,decor_by_subsampl,a_t);
            end
        end
    %end
%end