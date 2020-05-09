function new_neighbors = RNS(LSF, i_k, neighborhood, f_K, alfa)

     LSF_neighbors = LSF(:,neighborhood(:,i_k));
     LSF_i_k = LSF(:,i_k);
     q = repmat(LSF_i_k,[1, size(neighborhood(:, i_k), 1)]);
     p = LSF_neighbors;
     KLD_p = sum(p.* log2(p./q));
     KLD_q = sum(q.* log2(q./p));
     KLD = KLD_p + alfa * KLD_q;
     [~, ind] = sort(KLD);
     index = ind(1,1 : f_K);
     temp = neighborhood(:, i_k);
     
     new_neighbors = zeros(size(index, 2), 1);
     
     for i = 1 : size(index, 2)
         new_neighbors(i, 1) = temp(index(1,i));
     end
    
end