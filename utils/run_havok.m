function [A, B, eigA, Vh_havok] = run_havok(Vr, embedding_dim, model_rank, dt)
% Performs Hankel Alternative View of Koopman (HAVOK) analysis.
% NOTE: This is a basic implementation. Derivative estimation and regression might need refinement.
%
% Args:
%   Vr (double matrix): Truncated temporal coefficients (time x rank).
%   embedding_dim (int): Number of time delays (Hankel matrix rows, q).
%   model_rank (int): Rank of the linear system model (p < q).
%   dt (double): Time step.
%
% Returns:
%   A (double matrix): System matrix (p x p).
%   B (double vector): Input matrix (p x 1).
%   eigA (complex vector): Eigenvalues of A.
%   Vh_havok (double matrix): Right singular vectors of the Hankel matrix.

fprintf(' Running HAVOK...\n');
[n_samples, r] = size(Vr);

if model_rank >= embedding_dim
    error('HAVOK model rank (p=%d) must be less than embedding dimension (q=%d).', model_rank, embedding_dim);
end

% --- Select Time Series (typically the first PC) ---
ts = Vr(:, 1); 
fprintf(' Using first principal component (Vr(:,1)) for HAVOK.\n');

% --- Build Hankel Matrix ---
fprintf(' Building Hankel matrix with embedding dimension q=%d...\n', embedding_dim);
H = zeros(embedding_dim, n_samples - embedding_dim + 1);
for i = 1:embedding_dim
    H(i,:) = ts(i : n_samples - embedding_dim + i);
end
fprintf(' Hankel matrix H size: %d x %d.\n', size(H,1), size(H,2));

% --- SVD of Hankel Matrix ---
fprintf(' Performing SVD on Hankel matrix...\n');
[Uh, Sh, Vh_havok] = svd(H, 'econ'); % Vh_havok columns are the new coordinates

% --- Separate State and Forcing Variables ---
fprintf(' Separating state (p=%d) and forcing terms...\n', model_rank);
x = Vh_havok(:, 1:model_rank);   % State variables (time x p)
u = Vh_havok(:, model_rank + 1); % Assumed forcing term (time x 1)

% --- Estimate Derivatives (using simple finite difference) ---
% Central difference is generally better but needs careful boundary handling.
% Using forward difference for simplicity here (loses last point).
fprintf(' Estimating state derivatives using forward difference...\n');
dx_dt = zeros(size(x,1)-1, model_rank); 
for i = 1:model_rank
   dx_dt(:, i) = diff(x(:, i)) / dt; 
end
x_reg = x(1:end-1, :); % Match dimensions for regression
u_reg = u(1:end-1, :);

% --- Linear Regression (Solve dx/dt = Ax + Bu) ---
fprintf(' Performing linear regression to find A and B...\n');
Xi = [x_reg, u_reg]; % Regression matrix [x, u]
Theta = dx_dt;       % Target derivatives

% Solve Xi * [A; B'] = Theta  => [A; B'] = Xi \ Theta 
AB_stacked = Xi \ Theta; 

A = AB_stacked(1:model_rank, :);        % Extract A (p x p)
B = AB_stacked(model_rank + 1, :)';     % Extract B (p x 1) - note the transpose

fprintf(' Identified system matrices A (%dx%d) and B (%dx1).\n', size(A,1), size(A,2), size(B,1));

% --- Calculate Eigenvalues of A ---
fprintf(' Calculating eigenvalues of system matrix A...\n');
eigA = eig(A);

fprintf(' HAVOK complete.\n');

end