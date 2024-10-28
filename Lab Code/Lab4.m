% Load EEGLAB and electrode locations
eeglab; % Start EEGLAB
EEG.chanlocs = readlocs('/Users/lindseyma/Downloads/BCI.locs'); % Load electrode location file

% Define parameters
sampling_rate = 256; % Sampling rate in Hz (adjust if needed)
time_points = [0, 100, 200, 300]; % Time points in milliseconds
sample_points = round((time_points / 1000) * sampling_rate); % Convert to sample indices

% Assuming ERP is a matrix of (Electrodes x Time Points)
% Load the ERP data (already computed ERP should be in variable 'segments')
ERP = segments; % Replace with your actual ERP matrix variable

% Plot scalp maps for the specified time points
figure;
for i = 1:length(sample_points)
    subplot(2, 2, i); % 2x2 grid for 4 time points
    data = ERP(:, sample_points(i)); % Extract ERP data at the specific time point
    topoplot(data, EEG.chanlocs); % Plot the scalp map using topoplot
    title([num2str(time_points(i)) ' ms']); % Title indicating the time point
    colorbar; % Add color bar for visual reference
end
