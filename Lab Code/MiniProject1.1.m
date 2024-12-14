clear; close all;
% Load each dataset individually
train1_data = load('/Users/lindseyma/Downloads/MiniProject1_GroupData_Section2_2024/train_group5_section2.mat');
train2_data = load('/Users/lindseyma/Downloads/MiniProject1_GroupData_Section2_2024/train2_group5_section2.mat');
test1_data = load('/Users/lindseyma/Downloads/MiniProject1_GroupData_Section2_2024/test1_group5_section2.mat');
test2_data = load('/Users/lindseyma/Downloads/MiniProject1_GroupData_Section2_2024/test2_group5_section2.mat');

% Extract the 'y' matrix from each dataset
train1_data = train1_data.y;
train2_data = train2_data.y;
test1_data = test1_data.y;
test2_data = test2_data.y;

% Extract time, signals, onsets, and targets for train1_data
time_train1 = train1_data(1, :);
signals_train1 = train1_data(2:9, :); % EEG channels 1 to 8
onsets_train1 = train1_data(10, :);   % Flash onset markers
targets_train1 = train1_data(11, :);  % Target indicators

time_train2 = train2_data(1, :);
signals_train2 = train2_data(2:9, :); % EEG channels 1 to 8
onsets_train2 = train2_data(10, :);   % Flash onset markers
targets_train2 = train2_data(11, :);  % Target indicators

% Parameters for ERP segmentation
fs = 256;             % Sampling frequency
pre_onset = 100;      % 100 ms before onset
post_onset = 600;     % 600 ms after onset

% Convert time to samples
pre_samples = round(pre_onset / 1000 * fs);
post_samples = round(post_onset / 1000 * fs);

% Find onset indices for train1_data
onset_indices_train1 = find(onsets_train1 > 0); % Indices where flashes occur
target_indices_train1 = onset_indices_train1(targets_train1(onset_indices_train1) == 1); % Target flash indices
nontarget_indices_train1 = onset_indices_train1(targets_train1(onset_indices_train1) == 0); % Non-target flash indices

% Segment the data around each flash for target and non-target events
target_erp_train1 = [];
nontarget_erp_train1 = [];

% Extract segments for target events
for idx = target_indices_train1
    if idx - pre_samples > 0 && idx + post_samples <= length(signals_train1)
        segment = signals_train1(:, idx - pre_samples : idx + post_samples);
        target_erp_train1 = cat(3, target_erp_train1, segment); % Stack along 3rd dimension
    end
end

% Extract segments for non-target events
for idx = nontarget_indices_train1
    if idx - pre_samples > 0 && idx + post_samples <= length(signals_train1)
        segment = signals_train1(:, idx - pre_samples : idx + post_samples);
        nontarget_erp_train1 = cat(3, nontarget_erp_train1, segment); % Stack along 3rd dimension
    end
end

% Calculate the average ERP across all channels for target and non-target
avg_target_erp_train1 = mean(target_erp_train1, 3); % Average across trials (3rd dimension)
avg_nontarget_erp_train1 = mean(nontarget_erp_train1, 3); % Average across trials (3rd dimension)

% Time vector for plotting
time_vector = linspace(-pre_onset, post_onset, pre_samples + post_samples + 1);

% Plotting the ERP for train1_data
figure;
plot(time_vector, mean(avg_target_erp_train1, 1), 'b', 'LineWidth', 1.5); hold on;
plot(time_vector, mean(avg_nontarget_erp_train1, 1), 'r', 'LineWidth', 1.5);
title('Average ERP across all channels - Train1 Data');
xlabel('Time (ms)');
ylabel('Amplitude (\muV)');
legend('Target', 'Non-target');

% Marking ERP components (P1, N1, P2, P3/P300)
hold on;
plot([100 100], ylim, '--g', 'LineWidth', 1);  % P1 at 100ms
plot([200 200], ylim, '--m', 'LineWidth', 1);  % N1 at 200ms
plot([300 300], ylim, '--c', 'LineWidth', 1);  % P2 at 300ms
plot([400 400], ylim, '--k', 'LineWidth', 1);  % P300/P3 at 400ms

% Add text labels for the ERP components
text(100, max(ylim), 'P1', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', 'Color', 'g');
text(200, max(ylim), 'N1', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', 'Color', 'm');
text(300, max(ylim), 'P2', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', 'Color', 'c');
text(400, max(ylim), 'P300', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', 'Color', 'k');
hold off;

chanlocs = readlocs('/Users/lindseyma/Desktop/EEG/BCI.locs');

% Define the indices of channels used in this lab
used_channels = [1, 6, 11, 12, 13, 14, 15, 16];  % Corresponds to Fz, Cz, Pz, P3, P4, PO8, PO7, Oz

% Define the time points after onset for scalp maps
time_points = [0, 100, 200, 300];  % in ms

% Extract sample indices corresponding to the time points
sample_points = round((time_points + pre_onset) / 1000 * fs);

% Prepare data for scalp maps
% Averaged ERP across trials for target condition
scalp_data_target = avg_target_erp_train1;  % Averaged ERP data for target flashes
scalp_data_nontarget = avg_nontarget_erp_train1;  % Averaged ERP data for non-target flashes

% Plot scalp maps for each time point for Target Condition
figure;
for i = 1:length(time_points)
    subplot(2, 2, i);
    
    % Initialize a 16-element array with NaNs
    scalp_data = nan(1, 16);
    
    % Assign values to the used channels only
    scalp_data(used_channels) = scalp_data_target(:, sample_points(i));  % Target condition
    
    % Use topoplot to create the scalp map
    topoplot(scalp_data, chanlocs, 'maplimits', 'absmax');  % 'absmax' scales based on data range
    title(['Target ERP at ' num2str(time_points(i)) ' ms']);
    colorbar;
end

% Repeat for Non-Target Condition
figure;
for i = 1:length(time_points)
    subplot(2, 2, i);
    
    % Initialize a 16-element array with NaNs
    scalp_data = nan(1, 16);
    
    % Assign values to the used channels only
    scalp_data(used_channels) = scalp_data_nontarget(:, sample_points(i));  % Non-target condition
    
    % Use topoplot to create the scalp map
    topoplot(scalp_data, chanlocs, 'maplimits', 'absmax');  % 'absmax' scales based on data range
    title(['Non-Target ERP at ' num2str(time_points(i)) ' ms']);
    colorbar;
end

% Find onset indices for train2_data
onset_indices_train2 = find(onsets_train2 > 0); % Indices where flashes occur
target_indices_train2 = onset_indices_train2(targets_train2(onset_indices_train2) == 1); % Target flash indices
nontarget_indices_train2 = onset_indices_train2(targets_train2(onset_indices_train2) == 0); % Non-target flash indices

% Segment the data around each flash for target and non-target events
target_erp_train2 = [];
nontarget_erp_train2 = [];

% Extract segments for target events
for idx = target_indices_train2
    if idx - pre_samples > 0 && idx + post_samples <= length(signals_train2)
        segment = signals_train2(:, idx - pre_samples : idx + post_samples);
        target_erp_train2 = cat(3, target_erp_train2, segment); % Stack along 3rd dimension
    end
end

% Extract segments for non-target events
for idx = nontarget_indices_train2
    if idx - pre_samples > 0 && idx + post_samples <= length(signals_train2)
        segment = signals_train2(:, idx - pre_samples : idx + post_samples);
        nontarget_erp_train2 = cat(3, nontarget_erp_train2, segment); % Stack along 3rd dimension
    end
end

% Calculate the average ERP across all channels for target and non-target
avg_target_erp_train2 = mean(target_erp_train2, 3); % Average across trials (3rd dimension)
avg_nontarget_erp_train2 = mean(nontarget_erp_train2, 3); % Average across trials (3rd dimension)

% Time vector for plotting
time_vector = linspace(-pre_onset, post_onset, pre_samples + post_samples + 1);

% Plotting the ERP for train2_data
figure;
plot(time_vector, mean(avg_target_erp_train2, 1), 'b', 'LineWidth', 1.5); hold on;
plot(time_vector, mean(avg_nontarget_erp_train2, 1), 'r', 'LineWidth', 1.5);
title('Average ERP across all channels - Train2 Data');
xlabel('Time (ms)');
ylabel('Amplitude (\muV)');
legend('Target', 'Non-target');

% Marking ERP components (P1, N1, P2, P3/P300)
hold on;
plot([100 100], ylim, '--g', 'LineWidth', 1);  % P1 at 100ms
plot([200 200], ylim, '--m', 'LineWidth', 1);  % N1 at 200ms
plot([300 300], ylim, '--c', 'LineWidth', 1);  % P2 at 300ms
plot([400 400], ylim, '--k', 'LineWidth', 1);  % P300/P3 at 400ms

% Add text labels for the ERP components
text(100, max(ylim), 'P1', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', 'Color', 'g');
text(200, max(ylim), 'N1', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', 'Color', 'm');
text(300, max(ylim), 'P2', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', 'Color', 'c');
text(400, max(ylim), 'P300', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', 'Color', 'k');
hold off;

chanlocs = readlocs('/Users/lindseyma/Desktop/EEG/BCI.locs');

% Define the indices of channels used in this lab
used_channels = [1, 6, 11, 12, 13, 14, 15, 16];  % Corresponds to Fz, Cz, Pz, P3, P4, PO8, PO7, Oz

% Define the time points after onset for scalp maps
time_points = [0, 100, 200, 300];  % in ms

% Extract sample indices corresponding to the time points
sample_points = round((time_points + pre_onset) / 1000 * fs);

% Prepare data for scalp maps
% Averaged ERP across trials for target condition
scalp_data_target = avg_target_erp_train2;  % Averaged ERP data for target flashes
scalp_data_nontarget = avg_nontarget_erp_train2;  % Averaged ERP data for non-target flashes

% Plot scalp maps for each time point for Target Condition
figure;
for i = 1:length(time_points)
    subplot(2, 2, i);
    
    % Initialize a 16-element array with NaNs
    scalp_data = nan(1, 16);
    
    % Assign values to the used channels only
    scalp_data(used_channels) = scalp_data_target(:, sample_points(i));  % Target condition
    
    % Use topoplot to create the scalp map
    topoplot(scalp_data, chanlocs, 'maplimits', 'absmax');  % 'absmax' scales based on data range
    title(['Target ERP at ' num2str(time_points(i)) ' ms']);
    colorbar;
end

% Repeat for Non-Target Condition
figure;
for i = 1:length(time_points)
    subplot(2, 2, i);
    
    % Initialize a 16-element array with NaNs
    scalp_data = nan(1, 16);
    
    % Assign values to the used channels only
    scalp_data(used_channels) = scalp_data_nontarget(:, sample_points(i));  % Non-target condition
    
    % Use topoplot to create the scalp map
    topoplot(scalp_data, chanlocs, 'maplimits', 'absmax');  % 'absmax' scales based on data range
    title(['Non-Target ERP at ' num2str(time_points(i)) ' ms']);
    colorbar;
end

