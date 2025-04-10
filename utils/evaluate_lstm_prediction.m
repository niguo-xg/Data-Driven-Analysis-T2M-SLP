function [rmse_overall, rmse_per_component] = evaluate_lstm_prediction(YTest, YPred, num_features)
% Evaluates LSTM prediction performance using RMSE and plots results.
% Assumes YTest and YPred are denormalized matrices (features x time_steps).
%
% Args:
%   YTest (double matrix): True target values (denormalized, features x num_test_sequences).
%   YPred (double matrix): Predicted values (denormalized, features x num_test_sequences).
%   num_features (int): Number of features (PCA components).
%
% Returns:
%   rmse_overall (double): Overall Root Mean Squared Error.
%   rmse_per_component (double vector): RMSE for each individual feature/component.

fprintf(' Calculating RMSE...\n');
% --- Calculate RMSE ---
errors = YTest - YPred;
rmse_per_component = sqrt(mean(errors.^2, 2)); % RMSE for each component (row)
rmse_overall = sqrt(mean(errors(:).^2));       % Overall RMSE

fprintf(' Plotting prediction results for first few components...\n');
% --- Plotting ---
figure('Name', 'LSTM Prediction Results', 'Position', [200, 200, 1000, 600]);

num_plots = min(num_features, 4); % Plot first 4 components or fewer

for i = 1:num_plots
    subplot(num_plots, 1, i);
    plot(YTest(i, :), 'b-', 'LineWidth', 1.5);
    hold on;
    plot(YPred(i, :), 'r--', 'LineWidth', 1.5);
    hold off;
    ylabel(sprintf('PC %d', i));
    title(sprintf('Component %d: Prediction vs Actual (RMSE: %.4f)', i, rmse_per_component(i)));
    legend('Actual', 'Predicted', 'Location', 'northwest');
    grid on;
    if i == num_plots
       xlabel('Time Step (in test set)');
    end
    xlim([1, size(YTest, 2)]); % Adjust x-axis limit
end

sgtitle(sprintf('LSTM Prediction vs Actual Values (Overall RMSE: %.4f)', rmse_overall));
fprintf(' Plotting complete.\n');

end