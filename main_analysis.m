%% Main Analysis Script for ME5311 Project
clc; 
clear; 
close all;

fprintf('Starting main analysis script...\n');
addpath('utils');
% load("data\slp_workspace.mat");
% load("data\t2m_workspace.mat");

%% === Parameters ===
% --- Data Parameters ---
data_file = 'data/t2m.nc';       % Data file ('slp.nc' or 't2m.nc')
variable_name = 't2m';           % Variable name in the NetCDF file ('msl' or 't2m')
dt = 1;                          % Time step in days (assuming daily data)

% --- Analysis Parameters ---
pca_rank = 50;                   % Target rank for PCA truncation (CHOOSE WISELY based on SVD spectrum)
analysis_method = 'DMD';         % Choose analysis method: 'DMD' or 'HAVOK'

% --- HAVOK Specific Parameters (only used if analysis_method = 'HAVOK') ---
havok_embedding_dim = 100;       % Hankel matrix rows (q) - needs tuning
havok_model_rank = 15;           % Rank for HAVOK linear model (p < q) - needs tuning

% --- Plotting Parameters ---
num_pca_modes_to_plot = 4;       % How many PCA modes to visualize
num_dyn_modes_to_plot = 4;       % How many DMD/HAVOK related features to visualize
save_figures = false;            % Set to true to save figures
figure_save_path = './figures/'; % Folder to save figures
if save_figures && ~exist(figure_save_path, 'dir')
   mkdir(figure_save_path);
end

fprintf('Parameters set:\n Data file: %s\n Variable: %s\n PCA Rank: %d\n Method: %s\n', ...
        data_file, variable_name, pca_rank, analysis_method);

%% === 1. Load and Preprocess Data ===
fprintf('\nStep 1: Loading and Preprocessing Data...\n');
try
    [X_anomaly, t, lat, lon, mean_X, n_lat, n_lon] = load_and_preprocess(data_file, variable_name);
    [n_space, n_samples] = size(X_anomaly);
    fprintf('Data loaded. Shape (space x time): %d x %d\n', n_space, n_samples);
catch ME
    fprintf('Error loading or preprocessing data: %s\n', ME.message);
    return; % Stop script if loading fails
end

%% === 2. Perform PCA/SVD ===
fprintf('\nStep 2: Performing PCA/SVD...\n');
try
    [Ur, Sr, Vr, svals, explained_variance] = perform_pca(X_anomaly, pca_rank);
    fprintf('PCA complete. Using rank %d.\n', pca_rank);
    fprintf('Variance captured by rank %d: %.2f%%\n', pca_rank, sum(explained_variance(1:pca_rank))*100);
catch ME
    fprintf('Error during PCA: %s\n', ME.message);
    return; 
end

%% === 3. Plot PCA Results ===
fprintf('\nStep 3: Plotting PCA Results...\n');
try
    fig_pca = plot_pca_results(svals, explained_variance, Ur, lat, lon, n_lat, n_lon, num_pca_modes_to_plot);
    if save_figures
        saveas(fig_pca, fullfile(figure_save_path, 'pca_results.png'));
        fprintf('PCA results figure saved.\n');
    end
catch ME
    fprintf('Error plotting PCA results: %s\n', ME.message);
    % Continue analysis even if plotting fails
end

%% === 4. Perform Dynamic Analysis ===
fprintf('\nStep 4: Performing Dynamic Analysis using %s...\n', analysis_method);

dynamic_results = struct(); % Initialize struct to hold results

if strcmpi(analysis_method, 'DMD')
    try
        [lambda, Phi_phys, omega, growth_rate] = run_dmd(Vr, Ur, dt); % Pass Ur for physical modes
        dynamic_results.lambda = lambda;
        dynamic_results.Phi_phys = Phi_phys;
        dynamic_results.omega = omega;
        dynamic_results.growth_rate = growth_rate;
        fprintf('DMD analysis complete.\n');
    catch ME
        fprintf('Error during DMD analysis: %s\n', ME.message);
        return;
    end
    
elseif strcmpi(analysis_method, 'HAVOK')
    try
        [A, B, eigA, Vh_havok] = run_havok(Vr, havok_embedding_dim, havok_model_rank, dt);
        dynamic_results.A = A;
        dynamic_results.B = B;
        dynamic_results.eigA = eigA;
        dynamic_results.Vh_havok = Vh_havok; % Hankel SVD modes
         fprintf('HAVOK analysis complete.\n');
    catch ME
        fprintf('Error during HAVOK analysis: %s\n', ME.message);
        return;
    end
    
else
    fprintf('Error: Unknown analysis method "%s". Choose "DMD" or "HAVOK".\n', analysis_method);
    return;
end

%% === 5. Plot Dynamic Analysis Results ===
fprintf('\nStep 5: Plotting Dynamic Analysis Results...\n');
try
    fig_dyn = plot_dynamic_results(analysis_method, dynamic_results, ...
                                   lat, lon, n_lat, n_lon, num_dyn_modes_to_plot);
    if save_figures
        saveas(fig_dyn, fullfile(figure_save_path, sprintf('%s_results.png', lower(analysis_method))));
        fprintf('%s results figure saved.\n', analysis_method);
    end
catch ME
    fprintf('Error plotting dynamic results: %s\n', ME.message);
end

%% === 6. Prepare Data for LSTM ===
fprintf('\nStep 6: Preparing Data for LSTM...\n');

% --- Parameters for LSTM ---
train_ratio = 0.8;       % Percentage of data for training
sequence_length = 10;    % Input sequence length (e.g., use past 10 days to predict next day)

% --- Split Data ---
num_time_steps = size(Vr, 1);
split_idx = floor(train_ratio * num_time_steps);

Vr_train_raw = Vr(1:split_idx, :);
Vr_test_raw = Vr(split_idx+1:end, :);
fprintf('Data split: %d training samples, %d testing samples.\n', size(Vr_train_raw, 1), size(Vr_test_raw, 1));

% --- Prepare Sequences and Normalize ---
% The function will create input sequences (X) and target outputs (Y)
% and normalize data based on the training set statistics.
try
    [XTrain, YTrain, XTest, YTest, norm_params] = prepare_lstm_data(Vr_train_raw, Vr_test_raw, sequence_length);
    fprintf('LSTM data prepared. Sequence length: %d.\n', sequence_length);
    num_features = size(XTrain{1}, 1); % Should be pca_rank (r)
    fprintf('Number of features (PCA components): %d\n', num_features);
catch ME
    fprintf('Error preparing LSTM data: %s\n', ME.message);
    return;
end

%% === 7. Define LSTM Network Architecture ===
fprintf('\nStep 7: Defining LSTM Network Architecture...\n');

num_hidden_units = 100; % Number of hidden units in LSTM layer (Hyperparameter)
num_responses = num_features; % We predict all r components

layers = [ ...
    sequenceInputLayer(num_features, 'Name', 'Input')
    lstmLayer(num_hidden_units, 'OutputMode', 'last', 'Name', 'LSTM') % 'last' for sequence-to-one
    fullyConnectedLayer(num_responses, 'Name', 'FC')
    regressionLayer('Name', 'Output')];

fprintf('LSTM layers defined: Input(%d) -> LSTM(%d) -> FC(%d) -> Regression\n', ...
        num_features, num_hidden_units, num_responses);

%% === 8. Specify Training Options ===
fprintf('\nStep 8: Specifying Training Options...\n');

options = trainingOptions('adam', ...       % Optimizer
    'MaxEpochs', 50, ...                    % Maximum number of epochs (Hyperparameter)
    'MiniBatchSize', 64, ...                % Mini-batch size (Hyperparameter)
    'InitialLearnRate', 0.005, ...          % Initial learning rate (Hyperparameter)
    'GradientThreshold', 1, ...             % Gradient clipping threshold
    'Shuffle', 'every-epoch', ...           % Shuffle data every epoch
    'Plots', 'training-progress', ...       % Show training progress plot
    'Verbose', false);                      % Suppress iteration details in command window
    % 'ValidationData', {XValidation, YValidation}, ... % Optional: Use a validation set
    % 'ValidationFrequency', 10, ...                % Optional: Check validation every N iterations

%% === 9. Train LSTM Network ===

fprintf('\nConverting YTrain for training...\n');
try
    % --- Convert YTrain from cell to numeric matrix ---
    YTrain_matrix = cat(2, YTrain{:}); % Concatenate along the second dimension
    fprintf('Converted YTrain from cell array to numeric matrix (%d x %d).\n', ...
            size(YTrain_matrix, 1), size(YTrain_matrix, 2));

    % --- Optional but recommended: Check for NaN/Inf after conversion ---
    if any(isnan(YTrain_matrix(:))) || any(isinf(YTrain_matrix(:)))
        error('YTrain_matrix contains NaN or Inf values after conversion!');
    end
    YTrain_matrix = YTrain_matrix';

catch ME
    fprintf('Error converting YTrain to matrix or NaN/Inf detected: %s\n', ME.message);
    return;
end

fprintf('\nStep 9: Training LSTM Network...\n');
% This step requires the Deep Learning Toolbox and can take time.
try
    net = trainNetwork(XTrain, YTrain_matrix, layers, options);
    fprintf('LSTM training complete.\n');
catch ME
     fprintf('Error during LSTM training: %s\n', ME.message);
     fprintf('Ensure Deep Learning Toolbox is installed and licensed.\n');
     return;
end

%% === 10. Make Predictions on Test Set ===
fprintf('\nStep 10: Making Predictions on Test Set...\n');

try
    % Predict using the trained network
    % Note: predict processes sequence by sequence.
    YPred_normalized = predict(net, XTest); 
    
    % YPred_normalized will be (num_responses x num_test_sequences) if YTest is matrix
    % Or a cell array if YTest is cell array. Let's convert YTest back to matrix first.
    YTest_matrix_normalized = cat(2, YTest{:}); % If YTest was cell array

    % Denormalize predictions
    mu = norm_params.mu;
    sig = norm_params.sig;
    YPred_denormalized = YPred_normalized' .* sig + mu; % Denormalize predictions
    YTest_denormalized = YTest_matrix_normalized .* sig + mu; % Denormalize true test values
    
    fprintf('Predictions generated and denormalized.\n');
catch ME
    fprintf('Error during LSTM prediction or denormalization: %s\n', ME.message);
    return;
end

%% === 11. Evaluate and Visualize Predictions ===
fprintf('\nStep 11: Evaluating and Visualizing Predictions...\n');

try
    [rmse_overall, rmse_per_component] = evaluate_lstm_prediction(YTest_denormalized, YPred_denormalized, pca_rank);
    fprintf('Overall RMSE on test set: %.4f\n', rmse_overall);
catch ME
    fprintf('Error during evaluation/visualization: %s\n', ME.message);
end

fprintf('\nLSTM analysis finished.\n');
