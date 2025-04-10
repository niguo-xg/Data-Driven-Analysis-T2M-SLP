function fig_handle = plot_dynamic_results(analysis_method, results, lat, lon, n_lat, n_lon, num_modes_to_plot)
% Plots results from DMD or HAVOK analysis.
%
% Args:
%   analysis_method (char): 'DMD' or 'HAVOK'.
%   results (struct): Struct containing results from run_dmd or run_havok.
%   lat, lon (double vectors): Latitude and longitude coordinates.
%   n_lat, n_lon (int): Number of latitude/longitude points.
%   num_modes_to_plot (int): Number of dynamic modes/features to plot.
%
% Returns:
%   fig_handle: Handle to the generated figure.

fprintf(' Plotting %s results...\n', analysis_method);

fig_handle = figure('Name', [analysis_method ' Results'], 'Position', [150, 150, 1200, 800]);

if strcmpi(analysis_method, 'DMD')
    % --- DMD Plotting ---
    lambda = results.lambda;
    Phi_phys = results.Phi_phys;
    omega = results.omega;
    growth_rate = results.growth_rate;
    
    num_modes_avail = size(Phi_phys, 2);
    num_modes_to_plot = min(num_modes_to_plot, num_modes_avail);

    % Plot 1: DMD Eigenvalues
    subplot(2, ceil(num_modes_to_plot/2) + 1, 1); 
    plot(real(lambda), imag(lambda), 'bo', 'MarkerSize', 6, 'MarkerFaceColor', 'b');
    hold on;
    th = linspace(0, 2*pi, 100);
    plot(cos(th), sin(th), 'r--', 'LineWidth', 1); % Unit circle
    axis equal; grid on;
    xlabel('Real(\lambda)'); ylabel('Imag(\lambda)');
    title('DMD Eigenvalues');
    lim_val = max(1.1, max(abs(lambda))*1.1);
    xlim([-lim_val, lim_val]); ylim([-lim_val, lim_val]);
    
    % Identify modes to plot (e.g., by magnitude or frequency)
    [~, sort_idx] = sort(abs(abs(lambda)-1)); % Sort by distance to unit circle (persistence)
    % Alternative: sort by frequency magnitude, growth rate, etc.
    
    % Plot DMD Modes
    for i = 1:num_modes_to_plot
        k = sort_idx(i); % Get index of the i-th most persistent mode
        subplot(2, ceil(num_modes_to_plot/2) + 1, i + 1);
        
        mode_map = reshape(real(Phi_phys(:, k)), n_lon, n_lat); % Plot real part
        
        contourf(lon, lat, mode_map', 20, 'LineColor', 'none');
        set(gca, 'YDir', 'normal');
        colorbar;
        xlabel('Longitude'); ylabel('Latitude');
        title(sprintf('DMD Mode %d (Real Part)\nFreq=%.3f, Growth=%.3f', k, omega(k), growth_rate(k)));
        axis tight;
        clim_val = max(abs(mode_map(:))) * 0.8;
        if clim_val > 0
            caxis([-clim_val, clim_val]);
            colormap(gca, redblue);
        end
    end
    sgtitle('Dynamic Mode Decomposition Results');

elseif strcmpi(analysis_method, 'HAVOK')
    % --- HAVOK Plotting ---
    eigA = results.eigA;
    Vh_havok = results.Vh_havok; % (time x q)
    % A = results.A; % p x p
    % B = results.B; % p x 1
    
    [n_time_havok, q] = size(Vh_havok);
    p = length(eigA); % Model rank
    num_modes_to_plot = min(num_modes_to_plot, q); % Plot first few Vh modes
    
    % Plot 1: Eigenvalues of A
    subplot(2, ceil(num_modes_to_plot/2) + 1, 1);
    plot(real(eigA), imag(eigA), 'bo', 'MarkerSize', 6, 'MarkerFaceColor', 'b');
    grid on; axis equal;
    xlabel('Real(\lambda_A)'); ylabel('Imag(\lambda_A)');
    title('Eigenvalues of HAVOK System Matrix A');
    ax = gca; ax.XAxisLocation = 'origin'; ax.YAxisLocation = 'origin'; % Cross at zero
    
    % Plot HAVOK Coordinates (Columns of Vh)
    for k = 1:num_modes_to_plot
        subplot(2, ceil(num_modes_to_plot/2) + 1, k + 1);
        plot(1:n_time_havok, Vh_havok(:, k), 'LineWidth', 1);
        xlabel('Time Index (relative)'); 
        ylabel(sprintf('V_{%d}', k));
        title(sprintf('HAVOK Coordinate V_{%d}', k));
        if k == p+1
           title(sprintf('HAVOK Coordinate V_{%d} (Forcing u)', k)); 
           hold on; plot(xlim, [0 0], 'k--'); hold off; % Zero line for forcing
        elseif k <= p
            title(sprintf('HAVOK Coordinate V_{%d} (State x_{%d})', k, k));
        end
        grid on;
        xlim([1, n_time_havok]);
    end
     sgtitle('HAVOK Analysis Results');
     
else
    fprintf('Error: Plotting function does not recognize method "%s".\n', analysis_method);
    return;
end

fprintf(' %s plotting finished.\n', analysis_method);

end

% Helper colormap function (copied from plot_pca_results or use built-in)
function cmap = redblue()
    m = 64; r = linspace(0, 1, m/2)'; b = linspace(1, 0, m/2)'; g = r; 
    cmap = [[r; ones(m/2,1)], [g; flipud(g)], [ones(m/2,1); flipud(b)]];
end