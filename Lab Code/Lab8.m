clear; close all;

% Load data
data1 = load('/Users/lindseyma/Downloads/Lab8Data/lab8_p1_12hz.mat');
data2 = load('/Users/lindseyma/Downloads/Lab8Data/lab8_p1_20hz.mat');
data3 = load('/Users/lindseyma/Downloads/Lab8Data/lab8_p1_30hz.mat');
data4 = load('/Users/lindseyma/Downloads/Lab8Data/lab8_p2_10s_switch.mat');
data5 = load('/Users/lindseyma/Downloads/Lab8Data/lab8_p2_20hz.mat');
data6 = load('/Users/lindseyma/Downloads/Lab8Data/lab8_p2_30hz.mat');

% Assign variables
eeg_12hz = data1.ans;
eeg_20hz = data2.ans;
eeg_30hz = data3.ans;
eeg_10s_switch = data4.ans;
eeg_20hz_distracted = data5.ans;
eeg_30hz_distracted = data6.ans;

fs = 256; % Sampling frequency
discard_time = 10; % Discard first 10 seconds
discard_samples = fs * discard_time;

% Function to preprocess EEG data
function eeg_processed = preprocess_eeg(eeg_data, fs, discard_samples, duration, remove_channel)
    if remove_channel
        eeg_data(4, :) = []; % Remove the 4th channel (row)
    end
    eeg_data = eeg_data(:, discard_samples+1:end); % Discard the first 10 seconds (columns)
    keep_samples = fs * duration; % Number of samples to keep
    eeg_processed = eeg_data(:, 1:keep_samples); % Keep only the desired duration (columns)
end

% Preprocess datasets
eeg_12hz = preprocess_eeg(eeg_12hz, fs, discard_samples, 180, true); % 180s, remove 4th channel
eeg_20hz = preprocess_eeg(eeg_20hz, fs, discard_samples, 180, true); % 180s, remove 4th channel
eeg_30hz = preprocess_eeg(eeg_30hz, fs, discard_samples, 180, true); % 180s, remove 4th channel
eeg_10s_switch = preprocess_eeg(eeg_10s_switch, fs, discard_samples, 90, false); % 90s, keep all channels
eeg_20hz_distracted = preprocess_eeg(eeg_20hz_distracted, fs, discard_samples, 30, false); % 30s, keep all channels
eeg_30hz_distracted = preprocess_eeg(eeg_30hz_distracted, fs, discard_samples, 30, false); % 30s, keep all channels

% Function to compute FFT and plot
function plot_fft(eeg_data, fs, freq_label,stimulus_freq)
    L = size(eeg_data, 2); % Length of data (number of time points)
    NFFT = 2^nextpow2(L); % Zero-padding for better frequency resolution
    freq = (0:NFFT/2-1) * (fs / NFFT); % Frequency axis
    
    % FFT for all channels (excluding time if necessary)
    fft_data = fft(eeg_data', NFFT); % Transpose so time is along rows for FFT
    mag = abs(fft_data(1:NFFT/2, :)); % Take magnitude of positive frequencies
    
    % Average magnitude over all channels
    avg_mag = mean(mag, 2);
    
    % Exclude 0 Hz
    freq = freq(3:end);
    avg_mag = avg_mag(3:end);

    % Calculate and mark stimulus frequency
    freq_res = fs / NFFT; % Frequency resolution
    stimulus_index = round(stimulus_freq / freq_res) + 1; % Index of stimulus frequency

    
    % Plot average FFT
    figure;
    plot(freq, avg_mag);
    hold on;
    plot(freq(stimulus_index), avg_mag(stimulus_index), 'ro', 'MarkerSize', 8, 'LineWidth', 2); % Mark stimulus frequency
    title(['FFT Magnitude for ' freq_label]);
    xlabel('Frequency (Hz)');
    ylabel('Magnitude');
    legend('FFT Magnitude', ['Stimulus Frequency (' num2str(stimulus_freq) ' Hz)']);
    grid on;
    hold off;
end

% Compute and plot FFT for each condition
plot_fft(eeg_12hz, fs, '12 Hz Stimulus',12);
plot_fft(eeg_20hz, fs, '20 Hz Stimulus',20);
plot_fft(eeg_30hz, fs, '30 Hz Stimulus',30);

function display_channels_at_frequency(eeg_data, fs, stimulus_freq, freq_label)
    % eeg_data: EEG data matrix (channels x time)
    % fs: Sampling frequency
    % stimulus_freq: Modulator frequency (Hz)
    % freq_label: Title for the plot
    
    L = size(eeg_data, 2); % Length of data (number of time points)
    NFFT = 2^nextpow2(L); % Zero-padding to the next power of 2
    freq = (0:NFFT/2-1) * (fs / NFFT); % Frequency axis
    
    % FFT for all channels
    fft_data = fft(eeg_data', NFFT); % Transpose so time is along rows
    mag = abs(fft_data(1:NFFT/2, :)); % Magnitude for positive frequencies
    
    % Find the index of the stimulus frequency
    freq_res = fs / NFFT; % Frequency resolution
    stimulus_index = round(stimulus_freq / freq_res) + 1; % Index for the frequency
    
    % Extract magnitude at the stimulus frequency for all channels
    channel_magnitudes = mag(stimulus_index, :);
    
    % Plot the magnitude for all channels
    figure;
    bar(channel_magnitudes);
    title(['Channel Magnitudes at ' num2str(stimulus_freq) ' Hz (' freq_label ')']);
    xlabel('Channel Index');
    ylabel('Magnitude');
    grid on;
end

% Example Calls:
% Display channel magnitudes at the stimulus frequency for each condition
display_channels_at_frequency(eeg_12hz, fs, 12, '12 Hz Stimulus');
display_channels_at_frequency(eeg_20hz, fs, 20, '20 Hz Stimulus');
display_channels_at_frequency(eeg_30hz, fs, 30, '30 Hz Stimulus');

eeglab; % Start EEGLAB for channel location utilities
EEG.chanlocs = readlocs('/Users/lindseyma/Desktop/EEG/BCI.locs'); % Load channel locations

% Define relevant frequencies and stimuli magnitudes
frequencies = [12, 20, 30]; % Stimulus frequencies
stimuli_data = {eeg_12hz, eeg_20hz, eeg_30hz}; % EEG datasets
% Ensure selected electrodes exclude the discarded channel
selected_electrodes = {'Fz', 'Cz', 'Pz', 'P3', 'PO7', 'PO8', 'Oz'}; % Exclude the discarded 'P4'

% Find the indices of the selected electrodes
plotchans = find(ismember({EEG.chanlocs.labels}, selected_electrodes)); % Find the channels to plot

% Loop through each stimulus and create scalp maps
for i = 1:length(frequencies)
    % Extract EEG data and compute FFT
    eeg_data = stimuli_data{i}; % Already preprocessed to exclude the 4th channel
    L = size(eeg_data, 2); % Number of time points
    NFFT = 2^nextpow2(L); % Zero padding for FFT
    fft_data = fft(eeg_data', NFFT); % Compute FFT
    mag = abs(fft_data(1:NFFT/2, :)); % Magnitude for positive frequencies
    
    % Extract magnitude at the stimulus frequency
    freq_res = fs / NFFT; % Frequency resolution
    freq_idx = round(frequencies(i) / freq_res) + 1; % Index of the frequency
    avg_magnitudes = mean(mag(freq_idx, :), 1); % Average across remaining channels
    
    % Create a vector for scalp mapping
    assr_magnitudes = nan(1, length(EEG.chanlocs)); % Initialize with NaN
    assr_magnitudes(plotchans) = avg_magnitudes(:, 2:8); % Assign values only to selected electrodes
    
    % Create scalp map
    figure;
    topoplot(assr_magnitudes, EEG.chanlocs, 'maplimits', 'absmax', 'electrodes', 'labels', 'plotchans', plotchans);
    title(['Scalp Map at ' num2str(frequencies(i)) ' Hz']);
    colorbar;
end

% Parameters
strongest_eeg = eeg_12hz; % Strongest case (20 Hz stimulus)
stimulus_freq = 12; % Frequency of interest
interval_duration = 30; % Interval length in seconds
samples_per_interval = fs * interval_duration; % Number of samples per interval
total_samples = size(strongest_eeg, 2); % Total number of samples
num_intervals = floor(total_samples / samples_per_interval); % Number of 30s intervals

% Initialize storage for magnitudes
ssvep_magnitudes = zeros(1, num_intervals);

% Process each interval
for i = 1:num_intervals
    % Extract 30s interval
    start_idx = (i-1) * samples_per_interval + 1;
    end_idx = start_idx + samples_per_interval - 1;
    eeg_interval = strongest_eeg(:, start_idx:end_idx);
    
    % Compute FFT
    L = size(eeg_interval, 2);
    NFFT = 2^nextpow2(L);
    fft_data = fft(eeg_interval', NFFT); % FFT for each channel
    mag = abs(fft_data(1:NFFT/2, :)); % Magnitudes for positive frequencies
    
    % Extract magnitude at 20 Hz
    freq_res = fs / NFFT; % Frequency resolution
    freq_idx = round(stimulus_freq / freq_res) + 1; % Index of 20 Hz
    avg_mag = mean(mag(freq_idx, :), 2); % Average magnitude over channels
    
    % Store result
    ssvep_magnitudes(i) = avg_mag;
end

% Normalize magnitudes
normalized_magnitudes = ssvep_magnitudes / max(ssvep_magnitudes);

% Plot the results
figure;
plot((1:num_intervals) * interval_duration, normalized_magnitudes, '-o');
xlabel('Cumulative Signal Duration (s)');
ylabel('Normalized SSVEP Magnitude');
title('SSVEP Magnitude as a Function of Signal Duration');
grid on;

% Function for analyzing attention data
function attention_fft(eeg_data, fs, condition_label)
    L = size(eeg_data, 2); % Length of data (number of time points)
    NFFT = 2^nextpow2(L);
    freq = (0:NFFT/2-1) * (fs / NFFT);
    fft_data = fft(eeg_data', NFFT); % Transpose so time is along rows for FFT
    mag = abs(fft_data(1:NFFT/2, :));
    avg_mag = mean(mag, 2);

    % Exclude 0 Hz
    freq = freq(5:end);
    avg_mag = avg_mag(5:end);   

    % Plot FFT for the condition
    figure;
    plot(freq, avg_mag);
    title(['FFT Magnitude for Attention Condition: ' condition_label]);
    xlabel('Frequency (Hz)');
    ylabel('Magnitude');
    grid on;
end

% Analyze attention data
attention_fft(eeg_10s_switch, fs, '10s Switch Condition');
attention_fft(eeg_20hz_distracted, fs, '20Hz Distracted');
attention_fft(eeg_30hz_distracted, fs, '30Hz Distracted');

function analyze_specific_freqs(eeg_data, fs, target_freqs, label)
    L = size(eeg_data, 2); % Length of data
    NFFT = 2^nextpow2(L); % FFT length
    freq = (0:NFFT/2-1) * (fs / NFFT); % Frequency axis
    
    % Compute FFT
    fft_data = fft(eeg_data', NFFT); % Transpose for FFT
    mag = abs(fft_data(1:NFFT/2, :)); % Magnitude of positive frequencies
    
    % Find indices of target frequencies
    power_at_freqs = zeros(1, length(target_freqs));
    for i = 1:length(target_freqs)
        [~, idx] = min(abs(freq - target_freqs(i))); % Find closest frequency bin
        power_at_freqs(i) = mean(mag(idx, :)); % Average power at target frequency
    end
    
    % Display results
    disp(['Power at target frequencies for ', label, ':']);
    for i = 1:length(target_freqs)
        disp([num2str(target_freqs(i)), ' Hz: ', num2str(power_at_freqs(i))]);
    end
end

% Example usage:
analyze_specific_freqs(eeg_20hz_distracted, fs, [20, 30], '20Hz Distracted');
analyze_specific_freqs(eeg_30hz_distracted, fs, [20, 30], '30Hz Distracted');