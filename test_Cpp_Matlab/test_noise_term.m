function test_noise_term()
    dt = 0.05;
    n = 2;
    nb_pcl = 10000;
    pchol_cov_size = n*(n + 1);
    pchol_cov_noises = zeros( pchol_cov_size , pchol_cov_size );
    
    for i = 1:pchol_cov_size
        for j = 1:pchol_cov_size
            pchol_cov_noises(i,j) = cos(i*j);
        end
    end
    
    pchol_cov_noises
    
    bt = zeros(n,1,1,nb_pcl);
    
    for i = 1:n
        for j = 1:nb_pcl
            bt(i,1,1,j) = cos(i);
        end
    end
    
    bt(:,1)
    
    dim_noises = size(pchol_cov_noises,2);
    noises=pchol_cov_noises*randn(dim_noises,nb_pcl)*sqrt(dt);
%     noises=pchol_cov_noises*randn((n+1)*n,nb_pcl)*sqrt(dt);
    noises=permute(noises,[1 3 4 2]); % (n+1)*n x nb_pcl
    clear pchol_cov_noises; % (n+1)*n x 1 x 1 x nb_pcl
    theta_alpha0_dB_t = noises(1:n,1,1,:); % n(i) x 1 x 1 x nb_pcl
    alpha_dB_t =reshape(noises(n+1:end,1,1,:),[n n 1 nb_pcl]); % n(j) x n(i) x 1 x nb_pcl
    clear noises
    
    alpha_dB_t = bsxfun(@times,bt,alpha_dB_t); % m(j) x m(i) x 1 x nb_pcl
    alpha_dB_t = permute(sum(alpha_dB_t,1),[2 3 1 4]); % m(i) x 1 x 1 x nb_pcl
    
    d_b_m = alpha_dB_t + theta_alpha0_dB_t;% m(i) x 1 x 1 x nb_pcl
   
   
   mean(d_b_m,4)
end

