function fig_handle = plot_pca_results(svals, explained_variance, Ur, lat, lon, n_lat, n_lon, num_modes_to_plot)
% Plots PCA results: Singular value spectrum and spatial modes.
%
% Args:
%   svals (double vector): Singular values.
%   explained_variance (double vector): Variance explained by each mode.
%   Ur (double matrix): Truncated spatial modes (space x rank).
%   lat, lon (double vectors): Latitude and longitude coordinates.
%   n_lat, n_lon (int): Number of latitude/longitude points.
%   num_modes_to_plot (int): Number of spatial modes to plot.
%
% Returns:
%   fig_handle: Handle to the generated figure.

fprintf(' Plotting PCA results...\n');

rank_avail = size(Ur, 2);
num_modes_to_plot = min(num_modes_to_plot, rank_avail); % Don't plot more modes than available

fig_handle = figure('Name', 'PCA Results', 'Position', [100, 100, 1200, 600]);

% --- Plot 1: Singular Value Spectrum ---
subplot(2, ceil(num_modes_to_plot/2) + 1, 1); % Top-left or first plot
semilogy(1:length(svals), svals, 'o-', 'LineWidth', 1.5, 'MarkerSize', 4);
hold on;
semilogy(1:rank_avail, svals(1:rank_avail), 'ro', 'MarkerSize', 6); % Highlight truncated modes
hold off;
title('Singular Value Spectrum');
xlabel('Mode Number');
ylabel('Singular Value');
grid on;
xlim([0, length(svals)+1]);
legend('All Modes', 'Truncated Modes', 'Location', 'northeast');

% --- Plot 2: Cumulative Explained Variance ---
subplot(2, ceil(num_modes_to_plot/2) + 1, 2); % Second plot
cumulative_variance = cumsum(explained_variance);
plot(1:length(explained_variance), cumulative_variance * 100, 's-', 'LineWidth', 1.5, 'MarkerSize', 4);
hold on;
plot(1:rank_avail, cumulative_variance(1:rank_avail) * 100, 'rs', 'MarkerSize', 6); % Highlight truncated
plot([0 rank_avail], [cumulative_variance(rank_avail)*100 cumulative_variance(rank_avail)*100], 'r--'); % Line at cutoff
plot([rank_avail rank_avail], [0 cumulative_variance(rank_avail)*100], 'r--'); % Line at cutoff
hold off;
title('Cumulative Explained Variance');
xlabel('Mode Number');
ylabel('Variance Explained (%)');
grid on;
ylim([0 105]);
xlim([0 length(explained_variance)+1]);
text(rank_avail + 2, cumulative_variance(rank_avail)*100, sprintf('%.1f%%', cumulative_variance(rank_avail)*100), 'Color', 'r');

% --- Plot Spatial Modes ---
for k = 1:num_modes_to_plot
    subplot(2, ceil(num_modes_to_plot/2) + 1, k + 2); % Adjust subplot index
    
    mode_map = reshape(Ur(:, k), n_lon, n_lat); % Reshape back to spatial map
    
    contourf(lon, lat, mode_map', 20, 'LineColor', 'none'); % Use transpose '
    set(gca, 'YDir', 'normal'); % Ensure North is up
    colorbar;
    xlabel('Longitude');
    ylabel('Latitude');
    title(sprintf('PCA Mode %d (%.2f%% Var)', k, explained_variance(k)*100));
    axis tight; % Adjust axis limits
    clim_val = max(abs(mode_map(:))) * 0.8; % Symmetrize colorbar if appropriate
    if clim_val > 0
       caxis([-clim_val, clim_val]); 
       colormap(gca, redblue); % Use a diverging colormap
    end
end

sgtitle('Principal Component Analysis Results'); % Overall figure title
fprintf(' PCA plotting finished.\n');

end

% Helper colormap function (or use built-in if available)
function cmap = redblue()
    m = 64; % Number of colors
    r = linspace(0, 1, m/2)'; 
    b = linspace(1, 0, m/2)';
    g = r; 
    cmap = [[r; ones(m/2,1)], [g; flipud(g)], [ones(m/2,1); flipud(b)]];
end