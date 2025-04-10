function [lambda, Phi_phys, omega, growth_rate] = run_dmd(Vr, Ur, dt)
% Performs Dynamic Mode Decomposition (DMD) on the reduced data.
%
% Args:
%   Vr (double matrix): Truncated temporal coefficients (time x rank).
%   Ur (double matrix): Truncated spatial modes (space x rank).
%   dt (double): Time step between snapshots.
%
% Returns:
%   lambda (complex vector): DMD eigenvalues.
%   Phi_phys (complex matrix): DMD modes in the original physical space (space x rank).
%   omega (double vector): Frequencies of DMD modes (cycles per dt).
%   growth_rate (double vector): Growth/decay rates of DMD modes.

fprintf(' Running DMD...\n');
[n_samples, r] = size(Vr); % Vr is (time x rank)

% --- Create Snapshot Matrices ---
% We need (rank x time) for standard DMD formulation
X1 = Vr(1:end-1, :)';  % Shape: (rank x n_samples-1)
X2 = Vr(2:end, :)';    % Shape: (rank x n_samples-1)
fprintf(' Created snapshot matrices X1, X2 of size %d x %d.\n', size(X1, 1), size(X1, 2));

% --- Compute DMD Operator (using SVD projection) ---
fprintf(' Computing DMD operator via SVD projection...\n');
[Ux1, Sx1, Vx1] = svd(X1, 'econ');

% Check for small singular values which could cause instability
rank_dmd = r; % Start with full rank
tol = 1e-10; % Tolerance for singular values
if any(diag(Sx1) < tol * Sx1(1,1))
    rank_dmd = find(diag(Sx1) < tol * Sx1(1,1), 1) - 1;
    fprintf(' WARNING: Small singular values detected. Reducing DMD rank to %d.\n', rank_dmd);
    if rank_dmd == 0
        error('All singular values too small for stable DMD computation.');
    end
    Ux1 = Ux1(:, 1:rank_dmd);
    Sx1 = Sx1(1:rank_dmd, 1:rank_dmd);
    Vx1 = Vx1(:, 1:rank_dmd);
end

% Low-dimensional operator A_tilde
A_tilde = Ux1' * X2 * Vx1 / Sx1; 
fprintf(' Computed low-dimensional operator A_tilde (%d x %d).\n', size(A_tilde,1), size(A_tilde,2));

% --- Compute Eigenvalues and Eigenvectors of A_tilde ---
fprintf(' Computing eigenvalues and eigenvectors...\n');
[W, Lambda] = eig(A_tilde);
lambda = diag(Lambda); % DMD eigenvalues

% --- Compute DMD Modes ---
fprintf(' Computing DMD modes...\n');
% Modes in the reduced space (Projected Modes)
Phi = X2 * Vx1 / Sx1 * W; 

% Project DMD modes back to the original physical space
Phi_phys = Ur(:, 1:rank_dmd) * Phi; % Use only the Ur modes corresponding to the rank used
fprintf(' Computed physical DMD modes (space x rank_dmd): %d x %d.\n', size(Phi_phys,1), size(Phi_phys,2));

% --- Calculate Frequencies and Growth Rates ---
fprintf(' Calculating frequencies and growth rates...\n');
omega = angle(lambda) / (2 * pi * dt);  % Frequency in cycles per unit dt
growth_rate = log(abs(lambda)) / dt;   % Growth rate per unit dt

fprintf(' DMD complete.\n');

end