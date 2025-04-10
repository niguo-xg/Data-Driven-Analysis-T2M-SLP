function [XTrain, YTrain, XTest, YTest, norm_params] = prepare_lstm_data(data_train, data_test, sequence_length)
% Prepares data for LSTM training and testing: creates sequences and normalizes.
% Data is normalized based on the training set statistics (z-score).
%
% Args:
%   data_train (double matrix): Training data (time x features).
%   data_test (double matrix): Test data (time x features).
%   sequence_length (int): Length of input sequences.
%
% Returns:
%   XTrain (cell array): Training input sequences {features x seq_len}.
%   YTrain (cell array): Training target outputs {features x 1}.
%   XTest (cell array): Test input sequences {features x seq_len}.
%   YTest (cell array): Test target outputs {features x 1}.
%   norm_params (struct): Normalization parameters (mu, sig) from training data.

fprintf(' Normalizing data based on training set...\n');
% --- Normalize Data (Z-score based on training set) ---
mu = mean(data_train);
sig = std(data_train);
sig(sig == 0) = 1; % Avoid division by zero for constant features (if any)

data_train_normalized = (data_train - mu) ./ sig;
data_test_normalized = (data_test - mu) ./ sig;

norm_params = struct('mu', mu', 'sig', sig'); % Store as column vectors for broadcasting later

fprintf(' Creating input/output sequences...\n');
% --- Create Training Sequences ---
num_train_samples = size(data_train_normalized, 1);
XTrain = cell(num_train_samples - sequence_length, 1);
YTrain = cell(num_train_samples - sequence_length, 1); % Use cell array for consistency with trainNetwork input

for i = 1:(num_train_samples - sequence_length)
    input_seq = data_train_normalized(i : i + sequence_length - 1, :)'; % features x seq_len
    target_val = data_train_normalized(i + sequence_length, :)';       % features x 1
    XTrain{i} = input_seq;
    YTrain{i} = target_val;
end

% --- Create Test Sequences ---
num_test_samples = size(data_test_normalized, 1);
XTest = cell(num_test_samples - sequence_length, 1);
YTest = cell(num_test_samples - sequence_length, 1);

for i = 1:(num_test_samples - sequence_length)
    input_seq = data_test_normalized(i : i + sequence_length - 1, :)';
    target_val = data_test_normalized(i + sequence_length, :)';
    XTest{i} = input_seq;
    YTest{i} = target_val;
end

fprintf(' Sequence creation complete.\n');
fprintf(' Number of training sequences: %d\n', length(XTrain));
fprintf(' Number of test sequences: %d\n', length(XTest));

end