clear; close all;

% Load all runs and reshape the data
data1 = load('/Users/lindseyma/Downloads/Lab9Data/run1.mat');
data2 = load('/Users/lindseyma/Downloads/Lab9Data/run2.mat');
data3 = load('/Users/lindseyma/Downloads/Lab9Data/run3.mat');
data4 = load('/Users/lindseyma/Downloads/Lab9Data/run4.mat');
z1 = load('/Users/lindseyma/Downloads/classrun1.mat');
z2 = load('/Users/lindseyma/Downloads/classrun2.mat');
z3 = load('/Users/lindseyma/Downloads/classrun3.mat');
z4 = load('/Users/lindseyma/Downloads/classrun4.mat');

% Reshape the data for all runs
data1 = reshape(data1.y, [18, size(data1.y, 3)]);
data2 = reshape(data2.y, [18, size(data2.y, 3)]);
data3 = reshape(data3.y, [18, size(data3.y, 3)]);
data4 = reshape(data4.y, [18, size(data4.y, 3)]);
data = {data1, data2, data3, data4}; % Combine into a cell array
z = {z1.z1, z2.z2, z3.z3, z4.z4};   % Cue information for all runs

sampling_rate = 256;

% Define the bandpass filter parameters
low_cutoff = 8; % Low cutoff frequency in Hz
high_cutoff = 20; % High cutoff frequency in Hz
[b, a] = butter(2, [low_cutoff, high_cutoff] / (sampling_rate / 2), 'bandpass'); % 2nd-order Butterworth filter

% Initialize containers for trials across all runs
all_left_trials = {};
all_right_trials = {};

% Loop over all runs to segment trials
for run_idx = 1:length(data)
    % Extract current run data and cue matrix
    current_data = data{run_idx};
    current_z = z{run_idx}; % Use the correct cue matrix for each run
    
    % Extract trigger channel for the current run
    trigger_channel = current_data(18, :); % Row 18 is the trigger channel

    % Use the current cue matrix for classification
    left_indices = find(current_z(1, :) == 1); % Trials where `z(1, :)` is 1 (left trials)
    right_indices = find(current_z(2, :) == 1); % Trials where `z(2, :)` is 1 (right trials)

    % Segment the data for left trials
    for i = 1:length(left_indices)
        trial_idx = left_indices(i); % Get the trial index
        trial_start = (trial_idx - 1) * sampling_rate * 2; % Adjust based on trial onset
        trial_start = trial_start + 2 * sampling_rate; % 2 seconds offset for trigger
        trial_end = trial_start + diff([4.5, 8]) * sampling_rate; % 4.5-8s window

        % Check boundaries
        if trial_end > size(current_data, 2)
            continue; % Skip if out of range
        end

        % Extract trial data
        all_left_trials{end+1} = current_data(2:17, trial_start:trial_end); % Rows 2-17 are EEG channels
    end

    % Segment the data for right trials
    for i = 1:length(right_indices)
        trial_idx = right_indices(i); % Get the trial index
        trial_start = (trial_idx - 1) * sampling_rate * 2; % Adjust based on trial onset
        trial_start = trial_start + 2 * sampling_rate; % 2 seconds offset for trigger
        trial_end = trial_start + diff([4.5, 8]) * sampling_rate; % 4.5-8s window

        % Check boundaries
        if trial_end > size(current_data, 2)
            continue; % Skip if out of range
        end

        % Extract trial data
        all_right_trials{end+1} = current_data(2:17, trial_start:trial_end); % Rows 2-17 are EEG channels
    end
end

% Display total trials across all runs
fprintf('Total left trials across all runs: %d\n', length(all_left_trials));
fprintf('Total right trials across all runs: %d\n', length(all_right_trials));

% Initialize containers for average power
left_power = zeros(16, length(all_left_trials)); % Power for each electrode in each left trial
right_power = zeros(16, length(all_right_trials)); % Power for each electrode in each right trial

% Filter and calculate average power for left trials
for i = 1:length(all_left_trials)
    trial_data = all_left_trials{i}; % Get trial data
    filtered_data = filtfilt(b, a, trial_data); % Apply bandpass filter
    % Calculate average power for each electrode
    left_power(:, i) = mean(filtered_data .^ 2, 2); % Average power across time for each electrode
end

% Filter and calculate average power for right trials
for i = 1:length(all_right_trials)
    trial_data = all_right_trials{i}; % Get trial data
    filtered_data = filtfilt(b, a, trial_data); % Apply bandpass filter
    % Calculate average power for each electrode
    right_power(:, i) = mean(filtered_data .^ 2, 2); % Average power across time for each electrode
end

% Compute the overall average power for left and right conditions
mean_left_power = mean(left_power, 2); % Mean power for each electrode (left trials)
mean_right_power = mean(right_power, 2); % Mean power for each electrode (right trials)

% Display the results
disp('Average power for each electrode (Left trials):');
disp(mean_left_power);
disp('Average power for each electrode (Right trials):');
disp(mean_right_power);

% Visualize the power difference
power_difference = mean_left_power - mean_right_power; % Difference between conditions
figure;
bar(power_difference);
title('Power Difference (Left - Right)');
xlabel('Electrode Number');
ylabel('Power Difference');

% Start EEGLAB for channel location utilities
eeglab;

% Load channel locations
EEG.chanlocs = readlocs('/Users/lindseyma/Desktop/EEG/CSP.locs');

% Verify and select the 16 electrodes used in the experiment
selected_electrodes = {'FZ', 'F1', 'F2', 'FC5', 'FC3', 'FC1', ...
                       'FC2', 'FC4', 'FC6', 'C5', 'C1', 'C2', ...
                       'C6', 'CP3', 'CPz', 'CP4'}; % Match your 16 electrodes
plotchans = find(ismember({EEG.chanlocs.labels}, selected_electrodes)); % Indices of selected channels

% Normalize power_difference to match the 16 electrodes
% Ensure that the power_difference vector corresponds to `selected_electrodes`
power_difference_plot = power_difference(plotchans); % Match selected channels

% Plot the scalp map
figure;
topoplot(power_difference_plot, EEG.chanlocs, 'maplimits', 'maxmin', 'electrodes', 'on');
colorbar;
title('Scalp Map: Power Difference (Left - Right)');