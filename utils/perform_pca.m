function [Ur, Sr, Vr, svals, explained_variance] = perform_pca(X_anomaly, pca_rank)
% Performs Principal Component Analysis (PCA) using Singular Value Decomposition (SVD).
%
% Args:
%   X_anomaly (double matrix): Data matrix (space x time), mean removed.
%   pca_rank (int): The desired rank for truncation.
%
% Returns:
%   Ur (double matrix): Truncated spatial modes (POD modes) (space x rank).
%   Sr (double matrix): Truncated singular values (diagonal matrix) (rank x rank).
%   Vr (double matrix): Truncated temporal coefficients (PCs) (time x rank).
%   svals (double vector): All singular values.
%   explained_variance (double vector): Variance explained by each mode.

fprintf(' Performing SVD...\n');
[n_space, n_samples] = size(X_anomaly);

if pca_rank > min(n_space, n_samples)
    warning('PCA rank %d is larger than the maximum possible rank %d. Adjusting rank.', pca_rank, min(n_space, n_samples));
    pca_rank = min(n_space, n_samples);
end

% --- Perform SVD ---
% 'econ' is crucial for large datasets where n_space or n_samples is large
[U, S, V] = svd(X_anomaly, 'econ');
fprintf(' SVD complete.\n');

svals = diag(S);

% --- Calculate Explained Variance ---
explained_variance = svals.^2 / sum(svals.^2);

% --- Truncate ---
fprintf(' Truncating to rank %d...\n', pca_rank);
Ur = U(:, 1:pca_rank);
Sr = S(1:pca_rank, 1:pca_rank);
Vr = V(:, 1:pca_rank); % Note: V from svd is already (time x rank) when using 'econ' if space > time

fprintf(' PCA finished.\n');

end