function Y = RLMR(X, X_sl, param)

%% Robust Local Manifold Representation (RLMR)
% X: samples
% X_sl: spatial neighbors locations
% K: the number of neighbors before neighbor selection
% f_K: the final number of neighbors

K = param.K;
f_K = param.f_K;
d = param.d;
alfa = param.alfa;

[D, N] = size(X); 

% Global Data Normalizaiton (GDN)
G_mean = mean(X, 1);
G_std = std(X, 0, 1);
X = (X - repmat(G_mean, D, 1))./(repmat(G_std, D, 1));

% Local Data Normalization (LDN)
L_mean = mean(X, 2);
L_std = std(X, 0, 2);
X = (X - repmat(L_mean, 1, N))./(repmat(L_std, 1, N));

distance_temp = pdist(X_sl','cityblock');
distance_sl = squareform(distance_temp);

X_t = sum(X.^2, 1); 
distance= repmat(X_t, N, 1) + repmat(X_t', 1, N) - 2 * (X' * X); 

[~, index] = sort(distance); 
[~, index_sl] = sort(distance_sl); 

neighborhood = index(2:(1 + K),:); 
neighborhood_sl = index_sl(2:(1 + K),:); 

% Refined neighbor selection
% Construct local structure feature (LSF) using Eqs.(12) and (13) in the paper
LSF = zeros(K,N);
for ii = 1 : N
    X_neighbors = X(:, neighborhood(:, ii)); 
    X_ii_neighbors = [X(:, ii), X_neighbors];
    X_ii_neighbors_mean = mean(X_ii_neighbors, 2);
    X_ii_neighbors_std = std(X_ii_neighbors, 0, 2);
    X_ii_neighbors_normalized= (X_ii_neighbors - repmat(X_ii_neighbors_mean, 1, size(X_ii_neighbors, 2)))./ repmat(X_ii_neighbors_std, 1, size(X_ii_neighbors, 2));
    temp = repmat(X_ii_neighbors_normalized(:, 1), 1, size(X_neighbors, 2)) - X_ii_neighbors_normalized(:, 2 : size(X_ii_neighbors_normalized, 2));
    LSF(:, ii) = sum((exp(-(temp.^2))));
end

% Found and obtain the new neighbors using Eqs. (14-16)
neighborhood_new = zeros(f_K, N);
for i_k = 1 : N
    new_neighbors = RNS(LSF, i_k, neighborhood, f_K, alfa);
    neighborhood_new(:, i_k) = new_neighbors;
end

% Embedding the spatial contextual information
W = zeros(f_K, N); 
I = zeros(f_K, N);
J = repmat(1 : N, [f_K 1]);
for ii = 1 : N
    
    X_new_neighbors = X(:, neighborhood_new(:, ii));
    X_ii_new_neighbors = [X(:, ii), X(:, neighborhood_sl(1, ii)), X(:, neighborhood_sl(2, ii)), X(:,neighborhood_sl(3, ii)), X(:, neighborhood_sl(4, ii))];
    X_ii_spatial_spectral = [X_ii_new_neighbors, X_new_neighbors];
    X_ii_spatial_spectral_mean = mean(X_ii_spatial_spectral, 2);
    X_ii_spatial_spectral_std = std(X_ii_spatial_spectral, 0, 2);
    X_ii_spatial_spectral_normalized = (X_ii_spatial_spectral - repmat(X_ii_spatial_spectral_mean, 1, ...
             size(X_ii_spatial_spectral, 2)))./ repmat(X_ii_spatial_spectral_std, 1, size(X_ii_spatial_spectral, 2));

    B = X_ii_spatial_spectral_normalized(:, 6 : end);
    YB = [X_ii_spatial_spectral_normalized(:, 1); X_ii_spatial_spectral_normalized(:, 2); ...
          X_ii_spatial_spectral_normalized(:, 3); X_ii_spatial_spectral_normalized(:, 4); ...
          X_ii_spatial_spectral_normalized(:, 5)];

    Neig_B = [B; B; B; B; B];  
    Neig_Y = repmat(YB, 1, f_K);
    z = (Neig_B - Neig_Y);
    
    C = z' * z;
    C = C + eye(f_K, f_K) * 1e-6 * trace(C); % avoid the trival solution
    W(:, ii) = C \ ones(f_K, 1); % meet the sum-to-one constraint
    W(:, ii) = W(:, ii) / sum(W(:, ii));

    I(:,ii) = neighborhood_new(:,ii);
end

W_temp = sparse(J(:), I(:), W(:));
I_temp = sparse(1 : N, 1 : N, ones(1, N));

if mean2(size(W_temp) == size(I_temp)) == 1 % fast computation
    M = (I_temp - W_temp)'*(I_temp - W_temp);
else
    M = sparse(1 : N, 1 : N, ones(1, N), N, N, 4 * f_K * N);  
    for ii = 1 : N 
       w = W(:, ii);  
       jj = neighborhood_new(:, ii); 
       M(ii, jj) = M(ii, jj) - w'; 
       M(jj, ii) = M(jj, ii) - w; 
       M(jj, jj) = M(jj, jj) + w * w'; 
    end; 
end

% Calculation of embedding
options.disp = 0; 
options.isreal = 1; 
options.issym = 1;  
[Y, ~] = eigs(M, d + 1, 0, options); 
Y = Y(:, 1 : d)' * sqrt(N);
end





 