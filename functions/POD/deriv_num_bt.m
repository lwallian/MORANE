    function d_bt = deriv_num_bt(bt,dt)
        
        [N,nb_modes]=size(bt);
        if (N < 12 && floor((N+1)/2)==(N+1)/2) || N < 2
            error('There are not enough points');
        end
        
        % Filter for interrior
        FD13p = [ 1/5544 -3/1155 1/56 -5/63 15/56 -6/7 0 ...
            6/7 -15/56 5/63 -1/56 3/1155 -1/5544 ]';
        % Filters for the boundaries
        FD11p = [ -1/1260 5/504 -5/84 5/21 -5/6 0 5/6 -5/21 5/84 -5/504 1/1260 ]';
        FD9p = [ 1/280 -4/105 1/5 -4/5 0 4/5 -1/5 4/105 -1/280 ]';
        FD7p = [ -1/60  3/20 -3/4 0 3/4 -3/20 1/60 ]';
        FD5p = [ 1/12  -2/3 0 2/3 -1/12 ]';
        FD3p = [ -1/2 0 1/2 ]';
        FD2p = [ -1 1]';
        filters={FD2p FD3p FD5p FD7p FD9p FD11p FD13p};
        
        d_bt = zeros(size(bt)); % N x nb_modes
        
        % Boundaries
        
        % Boundaries pixels are not take into account
        
        % From 2 to 6 pixels far from the boundaries
        for k=2:min(6,N/2)
            % Get neighborhoods
            len_filter = 2*(k-1)+1;
            neighb1 = zeros([2 nb_modes len_filter ]);
            neighb1(1,:,:)= permute( bt(1:len_filter,:) ,[3 2 1]);
            neighb1(2,:,:)= permute( bt((end-len_filter+1):end,:) ,[3 2 1]);
            filter1=permute(filters{k},[2 3 1 ]); % 1 x 1 x len_filter
            db1 = bsxfun(@times,neighb1, filter1); % 2 x nb_modes x len_filter
            clear neighb1;
            db1= sum(db1,ndims(db1)); % 2 x nb_modes
            d_bt([k (N-(k-1)) ], :)= 1/dt * db1;
        end
        
        if N >= 13
            % Interior
            % Get neighborhoods
            len_filter = 13;
            neighb1 = get_neighborhood_mat_1D( bt, len_filter); %  N x nb_modes x len_filter
            neighb1 = neighb1(7:(end-6),:,:); % (N-12) x nb_modes x len_filter
            filter1=permute(filters{7},[2 3 1 ]); % 1 x 1 x len_filter
            db1 = bsxfun(@times,neighb1, filter1); % (N-12) x nb_modes x len_filter
            clear neighb1;
            db1= sum(db1,ndims(db1));  % (N-12) x nb_modes
            d_bt(7:(end-6),:)= 1/dt * db1 ;
        end
        
        
        function neighb = get_neighborhood_mat_1D(A,len)
            % A must have the size :  N x nb_modes
            %
            
            s=size(A);
            neighb = zeros([s len]);% ... x N x nb_modes x len
            
            le=ceil((len-1)/2);
            
            for i=-le:0
                neighb(1:end+i,:,le+1+i)=A(1-i:end,:);
            end
            for i =1:le
                neighb((1+i):end,:,le+1+i)=A(1:end-i,:);
            end
            
            neighb=neighb(:,:,end:-1:1);
            
        end
        
    end