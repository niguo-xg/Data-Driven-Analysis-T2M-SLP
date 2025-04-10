function [X_anomaly, t, lat, lon, mean_X, n_lat, n_lon] = load_and_preprocess(data_file, variable_name)
% Loads NetCDF data, reshapes it, and removes the temporal mean.
%
% Args:
%   data_file (char): Path to the NetCDF file.
%   variable_name (char): Name of the variable to load (e.g., 'msl', 't2m').
%
% Returns:
%   X_anomaly (double matrix): Preprocessed data (space x time), mean removed.
%   t (double vector): Time vector.
%   lat (double vector): Latitude vector.
%   lon (double vector): Longitude vector.
%   mean_X (double vector): Temporal mean of the spatial field.
%   n_lat (int): Number of latitude points.
%   n_lon (int): Number of longitude points.

fprintf(' Loading data from %s, variable %s...\n', data_file, variable_name);

% --- Load Data ---
try
    x = ncread(data_file, variable_name);
    t = ncread(data_file, 'time'); 
    lon = ncread(data_file, 'longitude'); 
    lat = ncread(data_file, 'latitude'); 
catch ME
    fprintf(' Error reading NetCDF file: %s\n', ME.message);
    rethrow(ME); % Pass the error up
end

% --- Get Dimensions ---
% Assuming data is (longitude, latitude, time) - ADJUST IF NECESSARY
[n_lon, n_lat, n_time] = size(x); 
n_space = n_lon * n_lat;

fprintf(' Original data dimensions: Lon=%d, Lat=%d, Time=%d\n', n_lon, n_lat, n_time);

% --- Reshape Data ---
% Reshape to (space, time) matrix
X_flat = reshape(x, n_space, n_time);
fprintf(' Reshaped data dimensions (space x time): %d x %d\n', n_space, n_time);

% --- Remove Temporal Mean ---
fprintf(' Removing temporal mean...\n');
mean_X = mean(X_flat, 2); % Calculate mean across time for each spatial point
X_anomaly = X_flat - mean_X;

fprintf(' Preprocessing complete.\n');

end