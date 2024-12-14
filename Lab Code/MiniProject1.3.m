clear; close all;
% Define base directory path
base_dir = '/Users/lindseyma/Downloads/MiniProject1_GroupData_Section2_2024 (1)/';

% Define group numbers (excluding your group, Group 5)
groups = [1, 2, 3, 6, 7];
fs = 256;
tmin = -100;         
tmax = 400;          
Nch = 8;
T = 128;
num_samples = round((tmax - tmin) / 1000 * fs);

% Initialize arrays for training and testing data
cross_subject_data = [];
cross_subject_labels = [];
other_groups_test_data = [];
other_groups_test_labels = [];

bp_filter = designfilt('bandpassiir', 'FilterOrder', 4, ...
                       'HalfPowerFrequency1', 0.5, 'HalfPowerFrequency2', 60, ...
                       'SampleRate', fs);

% Function to apply the bandpass filter to each trial in the data matrix
function filtered_data = apply_bandpass_filter(data_matrix, bp_filter, fs, Nch, num_samples)
    num_trials = size(data_matrix, 1);  % Number of trials
    filtered_data = zeros(size(data_matrix));  % Initialize filtered data matrix
    
    num_samples = 128;

    for trial = 1:num_trials
        % Reshape each trial to [Nch x num_samples] to separate channels and time samples
        trial_data = reshape(data_matrix(trial, :), Nch, num_samples);
        
        % Apply the filter to each channel separately
        for ch = 1:Nch
            trial_data(ch, :) = filtfilt(bp_filter, trial_data(ch, :));
        end
        
        % Reshape back to original format and store in filtered_data
        filtered_data(trial, :) = reshape(trial_data, 1, []);
    end
end

for g = groups
    % Load train1 and train2 data for the current group
    train1_path = fullfile(base_dir, ['train1_group' num2str(g) '_section2.mat']);
    train2_path = fullfile(base_dir, ['train2_group' num2str(g) '_section2.mat']);
    
    train1_data = load(train1_path);
    train2_data = load(train2_path);
    train1_data = train1_data.y;
    train2_data = train2_data.y;
    
    % Process train1 and train2 data using the `process_trials` function from the previous code
    [train1_matrix, train1_labels] = process_trials(train1_data, fs, tmin, tmax, Nch);
    [train2_matrix, train2_labels] = process_trials(train2_data, fs, tmin, tmax, Nch);
    
    train1_matrix = apply_bandpass_filter(train1_matrix, bp_filter, fs, Nch, num_samples);
    train2_matrix = apply_bandpass_filter(train2_matrix, bp_filter, fs, Nch, num_samples);

    % Append to cross-subject data
    cross_subject_data = [cross_subject_data; train1_matrix; train2_matrix];
    cross_subject_labels = [cross_subject_labels; train1_labels; train2_labels];
    
    % Load test1 and test2 data for the current group
    test1_path = fullfile(base_dir, ['test1_group' num2str(g) '_section2.mat']);
    test2_path = fullfile(base_dir, ['test2_group' num2str(g) '_section2.mat']);
    
    test1_data = load(test1_path);
    test2_data = load(test2_path);
    test1_data = test1_data.y;
    test2_data = test2_data.y;
    
    % Process test1 and test2 data
    [test1_matrix, test1_labels] = process_trials(test1_data, fs, tmin, tmax, Nch);
    [test2_matrix, test2_labels] = process_trials(test2_data, fs, tmin, tmax, Nch);
    test1_matrix = apply_bandpass_filter(test1_matrix, bp_filter, fs, Nch, num_samples);
    test2_matrix = apply_bandpass_filter(test2_matrix, bp_filter, fs, Nch, num_samples);

    % Append to other groups' test data
    other_groups_test_data = [other_groups_test_data; test1_matrix; test2_matrix];
    other_groups_test_labels = [other_groups_test_labels; test1_labels; test2_labels];
end

function [data_matrix, labels] = process_trials(data, fs, tmin, tmax, Nch)
    % Extract time, signals, onsets, and targets
    onsets = data(10, :);
    signals = data(2:9, :);
    targets = data(11, :);
    num_samples = 128;

    % Find onset indices
    onset_indices = find(onsets > 0);
    
    % Initialize matrix for trials and labels
    num_trials = length(onset_indices);
    data_matrix = zeros(num_trials, Nch * num_samples);
    %disp(Nch * num_samples);
    labels = zeros(num_trials, 1);
    
    for trial = 1:num_trials
        idx = onset_indices(trial);
        start_idx = idx - round(tmin / 1000 * fs);
        end_idx = start_idx + num_samples - 1;
        if start_idx > 0 && end_idx <= size(signals, 2)
            trial_data = [];
        
            % Extract segment for each channel
            for ch = 1:Nch
                segment = signals(ch, start_idx:end_idx);
                trial_data = [trial_data, segment];
            end
            
            if length(trial_data) == Nch * num_samples
                data_matrix(trial, :) = trial_data;  % Add to the data matrix
                labels(trial) = targets(idx);        % Assign the label for the trial
            end
        end
    end
end

% Train the cross-subject model using all combined training data
cross_subject_model = fitclinear(cross_subject_data, cross_subject_labels);

% Load and process Group 5's test data
test1_group5 = load('/Users/lindseyma/Downloads/MiniProject1_GroupData_Section2_2024 (1)/test1_group5_section2.mat');
test2_group5 = load('/Users/lindseyma/Downloads/MiniProject1_GroupData_Section2_2024 (1)/test2_group5_section2.mat');

[test1_matrix, test1_labels] = process_trials(test1_group5.y, fs, tmin, tmax, Nch);
[test2_matrix, test2_labels] = process_trials(test2_group5.y, fs, tmin, tmax, Nch);

% Combine test1 and test2 data
group5_test_data = [test1_matrix; test2_matrix];
group5_test_labels = [test1_labels; test2_labels];

% Predict and calculate accuracy
predictions_group5 = predict(cross_subject_model, group5_test_data);
accuracy_group5 = mean(predictions_group5 == group5_test_labels) * 100;

% Predict and calculate accuracy on other groups
predictions_other_groups = predict(cross_subject_model, other_groups_test_data);
accuracy_other_groups = mean(predictions_other_groups == other_groups_test_labels) * 100;

% Load model1_2 from the .mat file
load('model1_2.mat', 'model1_2');  % Loads model1_2 into the workspace

% Predict and calculate accuracy using within-subject model on other groups
predictions_within_subject_other_groups = predict(model1_2, other_groups_test_data);
accuracy_within_subject_other_groups = mean(predictions_within_subject_other_groups == other_groups_test_labels) * 100;

accuracy_within_subject_group5 = 73.806;

results_table = table({'Cross-Subject Model'; 'Within-Subject Model (Model1+2)'}, ...
                      [accuracy_group5; accuracy_within_subject_group5], ...
                      [accuracy_other_groups; accuracy_within_subject_other_groups], ...
                      'VariableNames', {'Model', 'GroupX_Test_Set_1_2', 'All_Other_Groups_Test'});

% Display the table
disp(results_table);

% Extract weights from the cross-subject model
weights_cross = cross_subject_model.Beta;

% Reshape weights to [Nch x T] to match the channel-time structure
weights_matrix_cross = reshape(weights_cross, Nch, T);

% Square the weights to get the power
weights_power_cross = weights_matrix_cross .^ 2;

% Average squared weights over time for each channel
weights_avg_cross_time = mean(weights_power_cross, 2);  % Result is Nch x 1

% Initialize a 16-element array with NaNs to fill unused channels for topoplot
scalp_weights_cross = nan(1, 16);
used_channels = [1, 6, 11, 12, 13, 14, 15, 16];  % Indices for Fz, Cz, Pz, P3, P4, PO8, PO7, Oz
scalp_weights_cross(used_channels) = weights_avg_cross_time;

% Load channel locations
chanlocs = readlocs('/Users/lindseyma/Desktop/EEG/BCI.locs');

% Plot using topoplot
figure;
topoplot(scalp_weights_cross, chanlocs, 'maplimits', 'absmax');
title('Cross-Subject Model: Average Squared Weights Across Time (Spatial Distribution)');
colorbar;

% Average squared weights over channels for each time sample
weights_avg_cross_channels = mean(weights_power_cross, 1);  % Result is 1 x T

% Define time vector for plotting (assuming -100 to 400 ms range)
time_vector = linspace(-100, 400, T);

% Plot the average squared weights over time
figure;
plot(time_vector, weights_avg_cross_channels, 'LineWidth', 1.5);
xlabel('Time (ms)');
ylabel('Average Squared Weights');
title('Cross-Subject Model: Average Squared Weights Over Time (Temporal Distribution)');