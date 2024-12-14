clear; close all;

% Load all runs dynamically and reshape
num_runs = 8; % Total number of runs
all_data = cell(1, num_runs);
all_z = cell(1, num_runs);

for run_idx = 1:num_runs
    % Load data
    data = load(sprintf('/Users/lindseyma/Downloads/MiniProjectData/run%d.mat', run_idx));
    all_data{run_idx} = reshape(data.y, [18, size(data.y, 3)]);
    
    % Load cue information (assume cue files follow the same pattern for run 1-4)
    if run_idx <= 4
        cue = load(sprintf('/Users/lindseyma/Downloads/classrun%d.mat', run_idx));
        all_z{run_idx} = cue.(sprintf('z%d', run_idx)); % Dynamically access z1, z2, etc.
    else
        % Reuse z1-z4 for runs 5-8 if no unique cue files are provided
        all_z{run_idx} = all_z{mod(run_idx-1, 4) + 1};
    end
end

% Sampling rate and bandpass filter
sampling_rate = 256;
low_cutoff = 8; % Hz
high_cutoff = 20; % Hz
[b, a] = butter(2, [low_cutoff, high_cutoff] / (sampling_rate / 2), 'bandpass'); % 2nd-order Butterworth filter
filter_params.b = b;
filter_params.a = a;

% Initialize containers for all trials
all_left_trials = {};
all_right_trials = {};

% Segment trials across all runs
for run_idx = 1:num_runs
    current_data = all_data{run_idx};
    current_z = all_z{run_idx};
    [left_trials, right_trials] = segment_trials(current_data, current_z, sampling_rate, filter_params);

    % Store segmented trials
    all_left_trials = [all_left_trials, left_trials];
    all_right_trials = [all_right_trials, right_trials];
end

% Combine trials into matrices (concatenate along the 3rd dimension)
left_matrix = cat(3, all_left_trials{:});
right_matrix = cat(3, all_right_trials{:});

% Compute CSP filters
CSP_filters = compute_csp_filters(left_matrix, right_matrix);

% Project data onto CSP filters
projected_left = reshape(CSP_filters' * reshape(left_matrix, size(left_matrix, 1), []), ...
                         size(CSP_filters, 2), size(left_matrix, 2), size(left_matrix, 3));
projected_right = reshape(CSP_filters' * reshape(right_matrix, size(right_matrix, 1), []), ...
                          size(CSP_filters, 2), size(right_matrix, 2), size(right_matrix, 3));

% Compute the mean projected features
mean_projected_left = squeeze(mean(projected_left, 2)); % Average over time
mean_projected_right = squeeze(mean(projected_right, 2)); % Average over time

% Plot the mean CSP projected features
figure;
plot(mean(mean_projected_left, 2), 'r', 'LineWidth', 2); hold on;
plot(mean(mean_projected_right, 2), 'b', 'LineWidth', 2);
legend('Left', 'Right');
title('Average CSP Projected Features');
xlabel('CSP Component');
ylabel('Mean Projected Feature');
grid on;

% Extract log-variance features
left_features = extract_features(projected_left);
right_features = extract_features(projected_right);

% Combine features and labels
data_all = [left_features, right_features];
labels = [ones(1, size(left_features, 2)), -ones(1, size(right_features, 2))];

% Cross-validation to calculate accuracy
num_iterations = 10;
train_ratio = 0.9;
[mean_accuracy, std_error] = classify_with_cross_validation(data_all, labels, num_iterations, train_ratio);
fprintf('Average accuracy: %.2f%% ± %.2f%%\n', mean_accuracy * 100, std_error * 100);

% Display CSP filters on scalp
eeglab; % Start EEGLAB
EEG.chanlocs = readlocs('/Users/lindseyma/Desktop/EEG/CSP.locs');
plot_csp_filters(CSP_filters, EEG.chanlocs);

% Plot variance before and after CSP
plot_variance_comparison(left_matrix, right_matrix, projected_left, projected_right);

function [left_trials, right_trials] = segment_trials(data, cue, sampling_rate, filter_params)
    b = filter_params.b;
    a = filter_params.a;

    % Identify left and right trials
    left_indices = find(cue(1, :) == 1);
    right_indices = find(cue(2, :) == 1);

    % Initialize containers
    left_trials = {};
    right_trials = {};

    % Segment left trials
    for i = 1:length(left_indices)
        trial_idx = left_indices(i);
        trial_start = (trial_idx - 1) * sampling_rate * 2 + 2 * sampling_rate + 1; % Start at 2s offset
        trial_end = trial_start + diff([4.5, 8]) * sampling_rate;

        if trial_end > size(data, 2), continue; end
        trial_data = data(2:17, trial_start:trial_end);
        filtered_data = filtfilt(b, a, trial_data');
        left_trials{end+1} = filtered_data';
    end

    % Segment right trials
    for i = 1:length(right_indices)
        trial_idx = right_indices(i);
        trial_start = (trial_idx - 1) * sampling_rate * 2 + 2 * sampling_rate + 1;
        trial_end = trial_start + diff([4.5, 8]) * sampling_rate;

        if trial_end > size(data, 2), continue; end
        trial_data = data(2:17, trial_start:trial_end);
        filtered_data = filtfilt(b, a, trial_data');
        right_trials{end+1} = filtered_data';
    end
end

function CSP_filters = compute_csp_filters(left_matrix, right_matrix)
    % Compute covariance matrices for left trials
    num_left_trials = size(left_matrix, 3);
    Cov_left = zeros(size(left_matrix, 1), size(left_matrix, 1), num_left_trials);
    for i = 1:num_left_trials
        Cov_left(:, :, i) = cov(left_matrix(:, :, i)'); % Covariance for each trial
    end
    Cov_left = mean(Cov_left, 3); % Average covariance matrix for left trials

    % Compute covariance matrices for right trials
    num_right_trials = size(right_matrix, 3);
    Cov_right = zeros(size(right_matrix, 1), size(right_matrix, 1), num_right_trials);
    for i = 1:num_right_trials
        Cov_right(:, :, i) = cov(right_matrix(:, :, i)'); % Covariance for each trial
    end
    Cov_right = mean(Cov_right, 3); % Average covariance matrix for right trials

    % Solve generalized eigenvalue problem
    [W, D] = eig(Cov_left, Cov_right);
    [~, sort_indices] = sort(diag(D), 'descend'); % Sort by eigenvalues
    CSP_filters = W(:, sort_indices(1:6)); % Select top 6 CSP filters
end

function features = extract_features(projected_data)
    features = squeeze(log(var(projected_data, 0, 2))); % Log-variance of CSP components
end

function [mean_accuracy, std_error] = classify_with_cross_validation(data_all, labels, num_iterations, train_ratio)
    accuracy_scores = zeros(1, num_iterations);

    for iter = 1:num_iterations
        % Balanced train-test split
        num_left = sum(labels == 1);
        num_right = sum(labels == -1);
        train_left = randperm(num_left, floor(train_ratio * num_left));
        train_right = randperm(num_right, floor(train_ratio * num_right));

        train_indices = [train_left, num_left + train_right];
        test_indices = setdiff(1:length(labels), train_indices);

        train_data = data_all(:, train_indices);
        test_data = data_all(:, test_indices);
        train_labels = labels(train_indices);
        test_labels = labels(test_indices);

        % Normalize features
        mean_features = mean(train_data, 2);
        std_features = std(train_data, 0, 2);
        train_data = (train_data - mean_features) ./ std_features;
        test_data = (test_data - mean_features) ./ std_features;

        % Train LDA
        lda_model = fitcdiscr(train_data', train_labels');
        predicted_labels = predict(lda_model, test_data');
        accuracy_scores(iter) = mean(predicted_labels == test_labels');
    end

    % Calculate mean and standard error of accuracy
    mean_accuracy = mean(accuracy_scores);
    std_error = std(accuracy_scores) / sqrt(num_iterations);
end

function plot_csp_filters(CSP_filters, chanlocs)
    for i = 1:size(CSP_filters, 2)
        figure;
        topoplot(CSP_filters(:, i), chanlocs, 'maplimits', 'maxmin', 'electrodes', 'on');
        colorbar;
        title(['CSP Filter ' num2str(i)]);
    end
end

function plot_variance_comparison(left_matrix, right_matrix, projected_left, projected_right)
    % Variance before CSP
    left_variance_raw = mean(var(left_matrix, 0, 2), 3);
    right_variance_raw = mean(var(right_matrix, 0, 2), 3);

    % Variance after CSP
    left_variance_csp = mean(var(projected_left, 0, 2), 3);
    right_variance_csp = mean(var(projected_right, 0, 2), 3);

    % Plot comparison
    figure;
    bar([left_variance_raw, right_variance_raw; left_variance_csp, right_variance_csp]');
    legend({'Left', 'Right'});
    title('Class Variance Before and After CSP');
    xlabel('Condition');
    ylabel('Variance');
    set(gca, 'XTickLabel', {'Raw', 'CSP'});
end

% Initialize accuracy storage for each run
run_accuracies = zeros(1, length(all_data));

% Loop through each run to calculate accuracy
for run_idx = 1:length(all_data)
    % Extract trials for this run
    current_data = all_data{run_idx};
    current_z = all_z{run_idx};
    [left_trials, right_trials] = segment_trials(current_data, current_z, sampling_rate, filter_params);

    % Combine trials into matrices
    left_matrix = cat(3, left_trials{:});
    right_matrix = cat(3, right_trials{:});

    % Project data onto CSP filters
    projected_left = reshape(CSP_filters' * reshape(left_matrix, size(left_matrix, 1), []), ...
                             size(CSP_filters, 2), size(left_matrix, 2), size(left_matrix, 3));
    projected_right = reshape(CSP_filters' * reshape(right_matrix, size(right_matrix, 1), []), ...
                              size(CSP_filters, 2), size(right_matrix, 2), size(right_matrix, 3));

    % Extract features (log-variance)
    left_features = extract_features(projected_left);
    right_features = extract_features(projected_right);

    % Combine features and labels
    data_all = [left_features, right_features];
    labels = [ones(1, size(left_features, 2)), -ones(1, size(right_features, 2))];

    % Train-test split
    train_ratio = 0.9;
    num_left = sum(labels == 1);
    num_right = sum(labels == -1);
    train_left = randperm(num_left, floor(train_ratio * num_left));
    train_right = randperm(num_right, floor(train_ratio * num_right));

    train_indices = [train_left, num_left + train_right];
    test_indices = setdiff(1:length(labels), train_indices);

    train_data = data_all(:, train_indices);
    test_data = data_all(:, test_indices);
    train_labels = labels(train_indices);
    test_labels = labels(test_indices);

    % Normalize features
    mean_features = mean(train_data, 2);
    std_features = std(train_data, 0, 2);
    train_data = (train_data - mean_features) ./ std_features;
    test_data = (test_data - mean_features) ./ std_features;

    % Train LDA
    lda_model = fitcdiscr(train_data', train_labels');
    predicted_labels = predict(lda_model, test_data');

    % Compute accuracy for this run
    run_accuracies(run_idx) = mean(predicted_labels == test_labels') * 100;
end

% Plot accuracy across runs
figure;
plot(run_accuracies, '-o', 'LineWidth', 2);
title('Accuracy Across Training Runs');
xlabel('Run Number');
ylabel('Classification Accuracy (%)');
grid on;