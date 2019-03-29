function plot_pdf_modes(param)
% Plot the probability distribution functiion for each mode over time

%% Parameters of the program
% type_data = 'DNS300_inc3d_3D_2017_04_02_blocks';
% type_data = 'turb2D_blocks';
type_data = param.type_data;
folder_results = param.folder_results;
nb_modes = param.nb_modes;

%% Loading the correct results file and important parameters included in it
results_file = [folder_results '2ndresult_' type_data '_' num2str(nb_modes) '_modes__a_cst__decor_by_subsampl_bt_decor_choice_auto_shanon_threshold_0.0005fct_test_b_fullsto.mat'];
% results_file = [folder_results '2ndresult_' type_data  num2str(nb_modes) '_modes__a_cst__decor_by_subsampl_bt_decor_choice_auto_shanon_threshold_0.0005fct_test_b_fullsto_modal_dt.mat'];

load(results_file)
[N_t, nb_modes, N_particules] = size(bt_MCMC);
bt_MCMC = mean(bt_MCMC,3); % taking the mean

%% Loop on modes
for i=1:nb_modes
    bi = bt_MCMC(:,i);
    fig = figure;
    output_file = [ folder_results 'pdf_' type_data '_' num2str(nb_modes) '_mode_number_' num2str(i) '.gif'];
    
    % Loop on time
    for t=1:N_t
        bit = bi(t,:)';
        [f, xit] = ksdensity(bit);
        plot(xit,f)
        xlim([min(min(bi))*1.2 max(max(bi))*1.2])
        ylim([0 max(f)*1.2])
        title(['Probability distribution of mode ' num2str(i) ' at t = ' num2str(t*param.dt)])
        xlabel('value')
        ylabel('pdf')
        drawnow
        frame = getframe(fig);
        im{t} = frame2im(frame);
        
        [A,map] = rgb2ind(im{t},256);
        if t == 1
            imwrite(A,map,output_file,'gif','LoopCount',Inf,'DelayTime',param.dt);
        else
            imwrite(A,map,output_file,'gif','WriteMode','append','DelayTime',param.dt);
        end
    end
    close;
end

end