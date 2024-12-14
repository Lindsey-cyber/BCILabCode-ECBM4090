clear; close all;
% Load each dataset individually
train1_data = load('/Users/lindseyma/Downloads/MiniProject1_GroupData_Section2_2024/train_group5_section2.mat');
train2_data = load('/Users/lindseyma/Downloads/MiniProject1_GroupData_Section2_2024/train2_group5_section2.mat');
test1_data = load('/Users/lindseyma/Downloads/MiniProject1_GroupData_Section2_2024/test1_group5_section2.mat');
test2_data = load('/Users/lindseyma/Downloads/MiniProject1_GroupData_Section2_2024/test2_group5_section2.mat');

% Extract the 'y' matrix from each dataset
train1_y = train1_data.y;
train2_y = train2_data.y;
test1_y = test1_data.y;
test2_y = test2_data.y;

% Define the number of samples in the window and number of channels
fs = 256;            % Sampling rate
tmin = -100;         % Start time in ms relative to flash
tmax = 400;          % End time in ms relative to flash
Nch = 8;             % Number of channels

% Calculate the number of samples in the time window
num_samples = round((tmax - tmin) / 1000 * fs);  % 179 samples

% Initialize matrices for each dataset
train1_matrix = []; % Store trials x features for train1
train2_matrix = []; % Store trials x features for train2
test1_matrix = [];  % Store trials x features for test1
test2_matrix = [];  % Store trials x features for test2

% Process each dataset
datasets = {train1_y, train2_y, test1_y, test2_y};
data_matrices = {train1_matrix, train2_matrix, test1_matrix, test2_matrix};

for d = 1:length(datasets)
    data = datasets{d}; % Current dataset
    onsets = data(10, :); % Flash onset markers
    signals = data(2:9, :); % EEG signals (channels 1 to 8)
    
    % Find onset indices
    onset_indices = find(onsets > 0); % Indices of flash events
    
    % Initialize data matrix for the current dataset
    num_trials = length(onset_indices);
    data_matrix = zeros(num_trials, Nch * num_samples); % Initialize trial x features matrix
    
    % Extract and store each trial in the matrix
    for trial = 1:num_trials
        idx = onset_indices(trial);
        
        % Check bounds to avoid indexing errors
        if idx - round(tmin / 1000 * fs) > 0 && idx + round(tmax / 1000 * fs) <= size(signals, 2)
            % Extract data from each channel and concatenate it into a single row
            trial_data = [];
            for ch = 1:Nch
                % Extract the segment for the current channel
                segment = signals(ch, idx - round(tmin / 1000 * fs) : idx - round(tmin / 1000 * fs) + num_samples - 1);
                trial_data = [trial_data, segment]; % Concatenate across channels
            end

            trial_mean = mean(trial_data);
            trial_std = std(trial_data);
            trial_data = (trial_data - trial_mean) / trial_std; % Z-score normalization
            
            data_matrix(trial, :) = trial_data; % Add to the data matrix
        end
    end
    
    % Store the data matrix in the corresponding cell array
    data_matrices{d} = data_matrix;
end

% After this loop, data_matrices will contain train1_matrix, train2_matrix, test1_matrix, test2_matrix
train1_matrix = data_matrices{1};
train2_matrix = data_matrices{2};
test1_matrix = data_matrices{3};
test2_matrix = data_matrices{4};

% Generate labels for each dataset based on processed onset indices
labels_matrices = cell(1, 4);  % To store labels for each dataset

for d = 1:length(datasets)
    data = datasets{d}; % Current dataset
    onsets = data(10, :); % Flash onset markers
    targets = data(11, :); % Target indicators (1 for target, 0 for non-target)
    
    % Find onset indices (flash events)
    onset_indices = find(onsets > 0); % Indices of flash events

    % Generate labels based on target indicators at onset indices
    labels = targets(onset_indices); % Directly use target values as labels

    % Store the labels array in the corresponding cell array
    labels_matrices{d} = labels;
end

% After this loop, labels_matrices will contain train1_labels, train2_labels, test1_labels, test2_labels
train1_labels = labels_matrices{1};
train2_labels = labels_matrices{2};
test1_labels = labels_matrices{3};
test2_labels = labels_matrices{4};

% Reshape test1_labels and test2_labels to be column vectors
train1_labels = train1_labels(:);  % Convert to 3600x1
train2_labels = train2_labels(:);  % Convert to 7920x1
test1_labels = test1_labels(:);  % Convert to 3600x1
test2_labels = test2_labels(:);  % Convert to 7920x1

% Combine train1 and train2 for Model 1+2
train1_2_matrix = [train1_matrix; train2_matrix];
train1_2_labels = [train1_labels; train2_labels];

% Combine test1 and test2 for combined testing
test1_2_matrix = [test1_matrix; test2_matrix];
test1_2_labels = [test1_labels; test2_labels];

% Train Model 1 (on train1 data only)
model1 = fitclinear(train1_matrix, train1_labels);

% Train Model 2 (on train2 data only)
model2 = fitclinear(train2_matrix, train2_labels);

% Train Model 1+2 (on both train1 and train2 data)
model1_2 = fitclinear(train1_2_matrix, train1_2_labels);

% Test Model 1
pred1_test1 = predict(model1, test1_matrix);
pred1_test2 = predict(model1, test2_matrix);
pred1_test1_2 = predict(model1, test1_2_matrix);

% Test Model 2
pred2_test1 = predict(model2, test1_matrix);
pred2_test2 = predict(model2, test2_matrix);
pred2_test1_2 = predict(model2, test1_2_matrix);

% Test Model 1+2
pred1_2_test1 = predict(model1_2, test1_matrix);
pred1_2_test2 = predict(model1_2, test2_matrix);
pred1_2_test1_2 = predict(model1_2, test1_2_matrix);

% Calculate accuracies for each model and test set
accuracy = @(y_true, y_pred) mean(y_true == y_pred) * 100;

% Accuracy results
accuracy_model1_test1 = accuracy(test1_labels, pred1_test1);
accuracy_model1_test2 = accuracy(test2_labels, pred1_test2);
accuracy_model1_test1_2 = accuracy(test1_2_labels, pred1_test1_2);

accuracy_model2_test1 = accuracy(test1_labels, pred2_test1);
accuracy_model2_test2 = accuracy(test2_labels, pred2_test2);
accuracy_model2_test1_2 = accuracy(test1_2_labels, pred2_test1_2);

accuracy_model1_2_test1 = accuracy(test1_labels, pred1_2_test1);
accuracy_model1_2_test2 = accuracy(test2_labels, pred1_2_test2);
accuracy_model1_2_test1_2 = accuracy(test1_2_labels, pred1_2_test1_2);

% Display accuracy table
results = table({'Model 1'; 'Model 2'; 'Model 1+2'}, ...
    [accuracy_model1_test1; accuracy_model2_test1; accuracy_model1_2_test1], ...
    [accuracy_model1_test2; accuracy_model2_test2; accuracy_model1_2_test2], ...
    [accuracy_model1_test1_2; accuracy_model2_test1_2; accuracy_model1_2_test1_2], ...
    'VariableNames', {'Model', 'Test_Set_1', 'Test_Set_2', 'Test_Set_1_2'});

disp(results);

weights = model1_2.Beta;  % Extract weights

% Reshape weights to Nch x T, where Nch = 8 and T = 179
T = 128;
weights_matrix = reshape(weights, Nch, T);

% Square the weights to get the power
weights_power = weights_matrix .^ 2;

% Average squared weights over time for each channel
weights_avg_over_time = mean(weights_power, 2) * 1e5;  % Result is Nch x 1
scalp_weights = nan(1, 16);

chanlocs = readlocs('/Users/lindseyma/Desktop/EEG/BCI.locs');

% Define the indices of channels used in this lab
used_channels = [1, 6, 11, 12, 13, 14, 15, 16];  % Corresponds to Fz, Cz, Pz, P3, P4, PO8, PO7, Oz
scalp_weights(used_channels) = weights_avg_over_time;

% Plot using topoplot for spatial distribution of average squared weights
figure;
topoplot(scalp_weights, chanlocs, 'maplimits', 'absmax');  % 'absmax' scales based on data range
title('Average Squared Weights Across Time (Spatial Distribution)');
colorbar;

% Average squared weights over channels for each time sample
weights_avg_over_channels = mean(weights_power, 1);  % Result is 1 x T

% Plot the average squared weights over time
time_vector = linspace(tmin, tmax, T);  % Time vector for the x-axis
figure;
plot(time_vector, weights_avg_over_channels);
xlabel('Time (ms)');
ylabel('Average Squared Weights');
title('Average Squared Weights Over Time (Temporal Distribution)');

% Combine train1 and train2 data and labels
all_train_data = [train1_matrix; train2_matrix];
all_train_labels = [train1_labels; train2_labels];

num_samples = T;  % Number of time samples per channel

% Separate target and non-target trials based on labels
target_trials = all_train_data(all_train_labels == 1, :);    % Target trials
nontarget_trials = all_train_data(all_train_labels == 0, :); % Non-target trials

% Reshape each trial matrix to [num_trials, num_channels, num_samples]
target_trials_reshaped = reshape(target_trials, [], Nch, num_samples);
nontarget_trials_reshaped = reshape(nontarget_trials, [], Nch, num_samples);

% Calculate grand average ERP over channels and trials
% Average over trials for each channel and time sample
avg_target_erp = mean(target_trials_reshaped, 1);       % [1, Nch, T]
avg_nontarget_erp = mean(nontarget_trials_reshaped, 1); % [1, Nch, T]

% Average over channels
grand_avg_target_erp = squeeze(mean(avg_target_erp, 2));    % Result is 1 x T
grand_avg_nontarget_erp = squeeze(mean(avg_nontarget_erp, 2)); % Result is 1 x T

% Define time vector for plotting
time_vector = linspace(-100, 600, T);  % From -100 ms to 600 ms

% Plot the grand average ERP for target and non-target conditions
figure;
plot(time_vector, grand_avg_target_erp, 'b', 'LineWidth', 1.5); hold on;
plot(time_vector, grand_avg_nontarget_erp, 'r', 'LineWidth', 1.5);
title('Grand Average ERP (Target vs. Non-Target)');
xlabel('Time (ms)');
ylabel('Amplitude (\muV)');
legend('Target', 'Non-Target');

% Marking ERP components (e.g., P1, N1, P2, P3/P300)
hold on;
plot([100 100], ylim, '--g', 'LineWidth', 1);  % P1 at ~100ms
plot([200 200], ylim, '--m', 'LineWidth', 1);  % N1 at ~200ms
plot([300 300], ylim, '--c', 'LineWidth', 1);  % P2 at ~300ms
plot([400 400], ylim, '--k', 'LineWidth', 1);  % P300 at ~400ms

% Add text labels for the ERP components
text(100, max(ylim), 'P1', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', 'Color', 'g');
text(200, max(ylim), 'N1', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', 'Color', 'm');
text(300, max(ylim), 'P2', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', 'Color', 'c');
text(400, max(ylim), 'P300', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', 'Color', 'k');
hold off;

% Define the time points after flash onset for scalp maps
time_points = [0, 100, 200, 300];  % in ms
sample_points = round((time_points + 100) / 1000 * fs);  % Indices relative to 0 ms (tmin)

% Average ERP across trials for each channel and each time sample
% Result is [Nch x T]
grand_avg_target_erp_by_channel = squeeze(mean(target_trials_reshaped, 1));
grand_avg_nontarget_erp_by_channel = squeeze(mean(nontarget_trials_reshaped, 1));

% Plot scalp maps for each specified time point for target condition
figure;
for i = 1:length(time_points)
    subplot(2, 2, i);
    
    % Get data for the current time point
    scalp_data = grand_avg_target_erp_by_channel(:, sample_points(i));
    
    % Initialize a 16-element array with NaNs to fill the unused channels
    scalp_data_full = nan(1, 16);
    used_channels = [1, 6, 11, 12, 13, 14, 15, 16];  % Indices for Fz, Cz, Pz, P3, P4, PO8, PO7, Oz
    scalp_data_full(used_channels) = scalp_data;  % Assign the data to the correct channels
    
    % Use topoplot to create the scalp map for target condition
    topoplot(scalp_data_full, chanlocs, 'maplimits', 'absmax');
    title(['Target ERP at ' num2str(time_points(i)) ' ms']);
    colorbar;
end

% Repeat for non-target condition
figure;
for i = 1:length(time_points)
    subplot(2, 2, i);
    
    % Get data for the current time point
    scalp_data = grand_avg_nontarget_erp_by_channel(:, sample_points(i));
    
    % Initialize a 16-element array with NaNs to fill the unused channels
    scalp_data_full = nan(1, 16);
    scalp_data_full(used_channels) = scalp_data;  % Assign the data to the correct channels
    
    % Use topoplot to create the scalp map for non-target condition
    topoplot(scalp_data_full, chanlocs, 'maplimits', 'absmax');
    title(['Non-Target ERP at ' num2str(time_points(i)) ' ms']);
    colorbar;
end

% Assuming model1_2 is already created
save('model1_2.mat', 'model1_2');  % Saves model1_2 to a .mat file