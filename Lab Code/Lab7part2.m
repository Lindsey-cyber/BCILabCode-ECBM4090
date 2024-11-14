clear; close all;
data1 = load('/Users/lindseyma/Downloads/Lab7Data/Lab7_group5_p1_attend_left.mat');
data2 = load('/Users/lindseyma/Downloads/Lab7Data/Lab7_group5_p1_attend_right.mat');
data3 = load('/Users/lindseyma/Downloads/Lab7Data/Lab7_group5_p1_attend_both.mat');
data4 = load('/Users/lindseyma/Downloads/Lab7Data/Lab7_group5_p1_attend_none.mat');

attend_left = data1.ans;
attend_right = data2.ans;
attend_both = data3.ans;
attend_none = data4.ans;

fs = 8000;  % Sampling frequency (adjust if different)
f1 = 730; f2 = 1130;  % Base frequencies
f1d = f1 + 40; f2d = f2 + 60;  % Slightly shifted frequencies for modulation

t = (0:fs*1-1)/fs;
td = (0:fs*1-1)/fs;

ind = [ones(1,240) zeros(1,180)];
ind1 = ind(randperm(length(ind)));
ind2 = ind(randperm(length(ind)));

wc1 = sin(2*pi*f1*t);     % Carrier for left channel
wd1 = sin(2*pi*f1d*td);   % Modulated tone for left channel
wc2 = sin(2*pi*f2*t);     % Carrier for right channel
wd2 = sin(2*pi*f2d*td);   % Modulated tone for right channel

w1 = []; w2 = [];
for cnt1 = 1:length(ind)
    if ind1(cnt1);  w1 = [w1 wc1]; else; w1 = [w1 wd1]; end
    if ind2(cnt1);  w2 = [w2 wc2]; else; w2 = [w2 wd2]; end
end

t = (1:length(w1))/fs;
wa1 = sin(2*pi*40*t) + 1.1;  % Modulation signal for left channel
wa2 = sin(2*pi*35*t) + 1.1;  % Modulation signal for right channel

w1 = w1 .* wa1;
w2 = w2 .* wa2;
w = [w1; w2];

figure;
subplot(2,1,1);
plot((1:fs*0.1)/fs, w1(1:fs*0.1)); % Plot first 100 ms for left channel
title('Left Channel - Time Waveform (First 100 ms)');
xlabel('Time (s)');
ylabel('Amplitude');

subplot(2,1,2);
plot((1:fs*0.1)/fs, w2(1:fs*0.1)); % Plot first 100 ms for right channel
title('Right Channel - Time Waveform (First 100 ms)');
xlabel('Time (s)');
ylabel('Amplitude');

figure;
subplot(2,1,1);
specgram(w1, 256, fs); % Spectrogram for left channel
title('Left Channel - Spectrogram');
xlabel('Time (s)');
ylabel('Frequency (Hz)');

subplot(2,1,2);
specgram(w2, 256, fs); % Spectrogram for right channel
title('Right Channel - Spectrogram');
xlabel('Time (s)');
ylabel('Frequency (Hz)');

% Load your datasets (assuming they are already loaded in the workspace)
datasets = {attend_left, attend_right, attend_both, attend_none};
conditions = {'Attend Left', 'Attend Right', 'Attend Both', 'Attend None'};

for i = 1:length(datasets)
    data = datasets{i};         % Get the current dataset
    condition = conditions{i};   % Get the corresponding condition name

    % Extract time and trigger signal
    time_data = data(1, :);       % Extract time from the 1st row
    trigger_signal = data(9, :);  % Extract trigger signal from the 9th row

    % Find indices where the trigger signal is 1 (indicating the response period)
    trigger_indices = find(trigger_signal == 1);

    % Extract the EEG data (rows 2 to 8) for only the triggered periods
    eeg_data_triggered = data(2:8, trigger_indices);

    % Initialize array to store FFT magnitudes
    N = length(trigger_indices);  % Length of the signal for FFT
    fft_magnitudes = zeros(size(eeg_data_triggered, 1), floor(N/2) + 1);
    
    % Loop over each channel to calculate the FFT
    for ch = 1:7
        % Perform FFT on the triggered data for this channel
        Y = fft(eeg_data_triggered(ch, :), N);
        
        % Compute the magnitude and take only the positive frequencies
        P2 = abs(Y / N);
        P1 = P2(1:floor(N/2) + 1);
        P1(2:end-1) = 2 * P1(2:end-1);  % Scale for single-sided spectrum
        
        % Store the magnitude of FFT for this channel
        fft_magnitudes(ch, :) = P1;
    end

    % Average the magnitude of FFT across all channels
    avg_fft_magnitude = mean(fft_magnitudes, 1);
    
    % Correct frequency vector calculation
    f = fs * (0:(N/2)) / N;

    % Plot the averaged FFT magnitude
    figure('Name', [condition ' - Average FFT Magnitude']);
    plot(f, avg_fft_magnitude);
    title(['Average FFT Magnitude - Condition: ', condition]);
    xlabel('Frequency (Hz)');
    ylabel('Magnitude');
    xlim([0, fs/2]);  % Limit x-axis to half the sampling rate
end

L = length(trigger_indices);  % Length of FFT, assume calculated from triggered data

% Modulator frequencies
mod_freqs = [35, 40];
num_channels = 7; % Number of EEG channels

% Initialize storage for average magnitudes
average_magnitudes = zeros(4, length(mod_freqs));

% Loop through each dataset and calculate average magnitudes
for cond_idx = 1:length(datasets)
    data = datasets{cond_idx};
    trigger_indices = find(data(9, :) == 1); % Get indices for triggered period
    eeg_data_triggered = data(2:8, trigger_indices);

    % FFT on each channel and calculate magnitude at modulator frequencies
    fft_magnitudes = zeros(num_channels, floor(L/2) + 1);
    
    for ch = 1:num_channels
        Y = fft(eeg_data_triggered(ch, :), L);
        P2 = abs(Y / L);
        P1 = P2(1:floor(L/2) + 1);
        P1(2:end-1) = 2 * P1(2:end-1);  % Single-sided FFT
        
        fft_magnitudes(ch, :) = P1;
    end

    % Frequency vector for plotting
    f = fs * (0:(L/2)) / L;
    
    % Find indices corresponding to 35 Hz and 40 Hz
    [~, idx_35] = min(abs(f - 35));
    [~, idx_40] = min(abs(f - 40));
    
    % Calculate average magnitude at 35 Hz and 40 Hz across channels
    average_magnitudes(cond_idx, 1) = mean(fft_magnitudes(:, idx_35)); % 35 Hz
    average_magnitudes(cond_idx, 2) = mean(fft_magnitudes(:, idx_40)); % 40 Hz
end

% Create a table to display the results
average_magnitude_table = array2table(average_magnitudes, ...
    'VariableNames', {'35Hz', '40Hz'}, ...
    'RowNames', conditions);

% Display the table
disp(average_magnitude_table);

% Plotting the results as a bar graph
figure;
bar(average_magnitudes);
set(gca, 'XTickLabel', conditions);
legend({'35 Hz', '40 Hz'}, 'Location', 'northeast');
title('Average Magnitude of ASSR Signal for Each Condition at Modulator Frequencies');
xlabel('Listening Condition');
ylabel('Average Magnitude');