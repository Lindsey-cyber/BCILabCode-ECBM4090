clear; close all;
fs = 44100;
fc = 900;
fm = 30;
d = 600;
t = (1 : d * fs) / fs;
wc = sin(2 * pi * fc * t);
wm = 1 + sin(2 * pi * fm * t);
w = wc .* wm;
delay = 15;
 
figure;
subplot(3,1,1);
plot(t(1:fs*0.1), w(1:fs*0.1));
subplot(3,1,2);
plot(t(1:fs*0.1), wm(1:fs*0.1));
subplot(3,1,3);
plot(t(1:fs*0.1), wc(1:fs*0.1));
title('Amplitude-Modulated Signal (First 100 ms)');
xlabel('Time (s)');
ylabel('Amplitude');
hold off;

figure;
subplot(1,3,1);
spectrogram(w(1:0.1*fs), 441, [], [], fs, 'yaxis');
subplot(1,3,2);
spectrogram(wm(1:0.1*fs), 441, [], [], fs,'yaxis');
subplot(1,3,3);
spectrogram(wc(1:0.1*fs), 441, [], [], fs,'yaxis');
colorbar;
hold off;

%pause(delay);
%soundsc(wm);

data = load('/Users/lindseyma/Downloads/Lab6Data/lab6_group2_part2.mat');
fs1 = 256;
sound_onset = delay * fs1;
sound_offset = sound_onset + fs1 * d - 1;
eeg = data.ans;
eeg_segment = eeg(:, sound_onset:sound_offset);
eeg_channels = eeg_segment(2:8, :);

L = size(eeg_channels, 2); % Number of samples in segment
f = (0:(L/2)) * (fs1 / L); % Frequency vector for positive frequencies
Y = zeros(size(eeg_channels,1), L/2 + 1);

for i = 1:size(eeg_channels,1)
    Y_full = abs(fft(eeg_channels(i, :)));
    Y(i, :) = Y_full(1:L/2 + 1); % Only keep positive frequencies
end

P1 = mean(Y,1);
P1_full = mean(Y_full,1);

figure;
plot(f, P1);
title('Mean Amplitude Spectrum across Channels');
xlabel('Frequency (Hz)');
ylabel('Amplitude');
hold on;
[~, idx] = min(abs(f - 30)); 
ylim_vals = ylim; 
plot([f(idx) f(idx)], ylim_vals, 'r:');
legend('Average Magnitude', 'Expected 40Hz Peak');
hold off;

P1_normalized = P1 / sum(P1,2);
P1_normalized_full = P1_full / sum(P1_full);

N = L/2 +1;
figure;
plot(linspace(0, fs1/2, N), P1_normalized);
title('Normalized One-Sided Spectrum across Channels');
xlabel('Frequency (Hz)');
ylabel('Relative Power');
hold on;
modulator_frequencies = [30, 40];
for mod_freq = modulator_frequencies
    [~, idx] = min(abs(f - mod_freq)); 
    plot(f(idx), P1_normalized(idx), 'ro'); 
end
legend('Normalized Spectrum', 'Modulator Frequencies (30 Hz, 40 Hz)');
hold off;

duration_interval = 30;
assr_frequency = 30;

assr_magnitudes = [];

% Find the index for the ASSR frequency in the normalized full spectrum
[~, idx_assr] = min(abs(f - assr_frequency)); % Closest index to ASSR frequency

samples_per_interval = duration_interval * fs1; 
for interval = samples_per_interval:samples_per_interval:L
    relative_magnitude = sum(P1_normalized_full(1:interval/2 + 1));
    
    % Extract the ASSR component magnitude from the cumulative spectrum
    assr_magnitude = relative_magnitude;
    assr_magnitudes = [assr_magnitudes, assr_magnitude];
end

% Plot the relative ASSR magnitude as a function of signal duration
time_intervals = (duration_interval:duration_interval:duration_interval * length(assr_magnitudes));
figure;
plot(time_intervals, assr_magnitudes, '-o');
title('Relative Magnitude of ASSR as a Function of Signal Duration');
xlabel('Signal Duration (s)');
ylabel('Relative Magnitude of ASSR (Normalized)');



eeglab; 
EEG.chanlocs = readlocs('/Users/lindseyma/Desktop/EEG/BCI.locs'); 
[~, idx_40Hz] = min(abs(f - assr_frequency));
all_electrodes = {EEG.chanlocs.labels};
selected_electrodes = {'Fz', 'Cz', 'Pz', 'FC1', 'FC2', 'CP1', 'CP2'};
assr_magnitudes_40Hz = zeros(1, length(all_electrodes));

for i = 1:length(selected_electrodes)
    % Calculate FFT for the i-th row in eeg_channels
    Y_full = abs(fft(eeg_channels(i, :)));
    
    % Normalize the spectrum
    Y_normalized = Y_full / sum(Y_full);
    
    % Store the normalized 40 Hz component for this electrode
    assr_magnitudes_40Hz(i) = Y_normalized(idx_40Hz);
end

plotchans = find(ismember({EEG.chanlocs.labels}, selected_electrodes));

figure;
topoplot(assr_magnitudes_40Hz, EEG.chanlocs, 'maplimits', [min(assr_magnitudes_40Hz) max(assr_magnitudes_40Hz)], ...
         'electrodes', 'on', 'style', 'map');
title('Relative Magnitude of ASSR at 40 Hz Across Selected Electrodes');
colorbar;
xlabel('Electrode Positions');
ylabel('Relative ASSR Magnitude (40 Hz)');