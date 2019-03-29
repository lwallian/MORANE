function super_main_fv()
% Launch a set of simulations with a several set of parameters
% Especially several number of modes
%

init;
igrida=false;
% igrida=true;


% type_data = 'incompact3D_noisy2D_40dt_subsampl'
% type_data = 'inc3D_Re300_40dt_blocks'
% type_data = 'inc3D_HRLESlong_Re3900_blocks'
% type_data = 'DNS300_inc3d_3D_2017_04_02_blocks'
% type_data = 'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks'
% type_data = 'DNS300_inc3d_3D_2017_04_09_NOT_BLURRED_blocks'
%type_data = 'DNS300_inc3d_3D_2017_04_09_blocks'
type_data = 'DNS100_inc3d_2D_2017_04_29_blocks'


switch type_data
        
    case {'incompact3D_noisy2D_40dt_subsampl','inc3D_Re300_40dt_blocks'}
        v_nb_mode = 2
        a_t = false
        v_threshold=0.0005
        
    case {'DNS100_inc3d_2D_2017_04_29_blocks'}
%         v_nb_mode = 2
%         a_t = false
%         v_threshold=0.0005 
        
        v_nb_mode = 2.^(2:-1:1)
        a_t = [ true false]
        % v_threshold= [5e-4 1e-4  1e-5] % 
        v_threshold= [1e-2 1e-3 ] % 
        
    case {'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks'}
%         v_nb_mode = 2
%         a_t = false
%         v_threshold=0.0005 % trop petit -> diffusion trop faible
%         % v_threshold=4e-4 % trop petit -> diffusion trop faible
        
        v_nb_mode = 2.^(1)
        a_t = [true ]
        v_threshold= [ 1e-4] % 
        
%         v_nb_mode = 2.^(2:-1:1)
%         a_t = [ false]
%         v_threshold= [5e-4 4e-4 1e-4] % 
        
    case {'DNS300_inc3d_3D_2017_04_09_NOT_BLURRED_blocks'}
% %         v_nb_mode = 2.^(2:-1:1)
% %         a_t = [true false]
% %         v_threshold= [5e-4 4e-4 1e-4] % 
%         v_threshold=0.0005 % 
%         v_threshold=4e-4 % 
        
        v_nb_mode = 2.^(2:-1:1)
         a_t = [true false]
        % a_t = [ false]
        v_threshold= [1e-2] % 
        %v_threshold= [1e-2 1e-3 1e-4] % 
        % v_threshold= [5e-4 4e-4] % 
        
    case {'DNS300_inc3d_3D_2017_04_09_blocks'}
        v_nb_mode = 2.^(2:-1:1)
        % v_nb_mode = 2.^(1:2)
        %v_threshold= [5e-4 1e-4] % 
        v_threshold= [1e-2 1e-3 1e-4 1e-5] % 
         a_t = [true false]
         
%         v_nb_mode = 2
%         a_t = false
%         
%         v_threshold=0.0005 % trop grand -> diffusion trop grande
%         v_threshold=4e-4 % trop grand -> diffusion trop grande
%         v_threshold=1e-4 % trop petit -> diffusion trop faible
%         v_threshold=3e-4 % trop grand -> diffusion trop grande
%         v_threshold=2e-4 % trop grand -> diffusion trop grande
%         v_threshold=1.5e-4 % PAS GOOD -> bonne amplitude mais pas la bonne frequence
        
    case {'DNS300_inc3d_3D_2017_04_02_blocks'}
        v_nb_mode = 2;
%         v_nb_mode = 2.^(2:-1:1);
        a_t = false
        v_threshold=1e-3 
        % v_threshold=5e-4 % trop petit -> diffusion trop faible
        % % v_threshold=5e-4 % trop grand -> diffusion trop forte
        % v_threshold=1e-6 % trop petit -> diffusion trop faible
        % v_threshold=1e-4 % trop petit -> diffusion trop faible
        % v_threshold=2e-4 % trop petit -> diffusion trop faible
        % v_threshold=4e-4 % trop petit -> diffusion trop faible
        
    case 'inc3D_HRLESlong_Re3900_blocks'
        v_nb_mode = 2.^(4:-1:1);
%         v_nb_mode = 2.^(1:4);
        a_t = [true false]
        v_threshold=[10 1 0.1]/1000
end
%%

coef_correctif_estim.learn_coef_a=false;
save_all_bi=false;
decor_by_subsampl.bool=true;
%     cell_test_fct={'b'};
decor_by_subsampl.test_fct='b';
save_all_bi=false;
%     %     cell_test_fct={'b' 'db'};
%     %     v_threshold=[1 10]/1000;
%     %     v_threshold=1e-3;
%     %   v_threshold=[0.5 5]/1000;

%     v_threshold=[1 10]/1000; for LES 3900
%     v_threshold=0.0005; % for inc DNS 300 episode3
%     N_estim = [ 1 4 ]*842; % for inc DNS 300 episode3
N_estim = nan; % for inc DNS 3900
%          N_estim = [ 1 9 ]*250; % for inc DNS 3900
%     v_threshold=[0.1 0.5 1 5 10]/1000;
for j =1:length(a_t)
    %         for i=1:length(cell_test_fct)
    decor_by_subsampl.meth='bt_decor';
    decor_by_subsampl.choice_n_subsample='auto_shanon'; % need 'bt_decor' or  'a_estim_decor'
    for q=1:length(v_threshold)
        for k=v_nb_mode
            %             for k=2:2:nb_mode_max
            for l=1:length(N_estim)
                decor_by_subsampl.n_subsampl_decor=nan;
                decor_by_subsampl.spectrum_threshold=v_threshold(q);
                %                 decor_by_subsampl.test_fct=cell_test_fct{i}
                main_finite_variation(k,igrida, ...
                    coef_correctif_estim,save_all_bi, ...
                    decor_by_subsampl,type_data,a_t(j));
                %                         decor_by_subsampl,a_t(j),N_estim(l));
            end
        end
    end
end
% end