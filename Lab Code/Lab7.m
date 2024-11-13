clear; close all;
data1 = load('/Users/lindseyma/Downloads/Lab7Data/lab7_group5_p2_35hz_1107.mat');
data2 = load('/Users/lindseyma/Downloads/Lab7Data/lab7_group5_p2_40hz.mat');

p2_35hz = data1.ans;
p2_40hz = data2.ans;

fs = 256; 
%target_freq = 40;
time = p2_35hz(1,:);
eeg_p2_35hz = p2_35hz(2,:);
eeg_p2_40hz = p2_40hz(2,:);

eeg_p2_35hz_cut = eeg_p2_35hz;
eeg_p2_35hz_cut(1:2561) = NaN;

figure;
subplot(2,1,1);
plot(time,eeg_p2_35hz_cut);
title('EEG Amplitude 35Hz trial');
xlabel('Time (s)');
ylabel('EEG Amplitude (µV)');
subplot(2,1,2);
plot(time,eeg_p2_40hz);
title('EEG Amplitude 40Hz trial');
xlabel('Time (s)');
ylabel('EEG Amplitude (µV)');

t_start = 100; 
t_end = 300; 

trigger_signal = p2_35hz(3, :); 
index = find(trigger_signal == 1, 1); 

start_sample = t_start * fs;
end_sample = t_end * fs - 1;

eeg_35hz = p2_35hz(2, start_sample:end_sample); 
eeg_40hz = p2_40hz(2, start_sample:end_sample); 

%eeg = eeg - mean(eeg);
L = size(eeg_40hz, 2);
f = (0:(L/2)) * (fs / L);

average_amplitude_35hz = mean(eeg_35hz);
average_amplitude_40hz = mean(eeg_40hz);
disp(['Average amplitude of the envelope at 35 Hz from 100 to 300 seconds: ', num2str(average_amplitude_35hz)]);
disp(['Average amplitude of the envelope at 40 Hz from 100 to 300 seconds: ', num2str(average_amplitude_40hz)]);

data3 = load('/Users/lindseyma/Downloads/Lab7Data/lab7_group5_p3.mat');

%t_start2 = 60;
%start_sample2 = t_start2 * fs;
eeg_ratio = zeros(5, 60*fs);
threshold = zeros(1, 5);

for i = 1:5
    start_index = (i - 1) * 60 * fs + 1;
    end_index = i * 60 * fs;
    eeg_ratio(i,:) = data3.ans(2, start_index:end_index);
    threshold(i) = mean(eeg_ratio(i,:));
end

right_threshold = mean(threshold(mod(2:5,2)==0));
left_threshold = mean(threshold(mod(2:5,2)==1));
disp(['right_threshold: ', num2str(right_threshold)]);
disp(['left_threshold: ', num2str(left_threshold)]);

start = 60 * fs;
eeg_ratio_raw = data3.ans(2, :); 
eeg_trigger = data3.ans(3,:);
time_ratio = (0:length(eeg_ratio_raw) - 1) / fs + 60;

figure;
plot(time_ratio, eeg_ratio_raw);
hold on;
yline(right_threshold, 'r--', 'Right Threshold');
yline(left_threshold, 'b--', 'Left Threshold');
xlabel('Time (s)');
ylabel('EEG Ratio');
title('EEG Ratio over Time with Left and Right Thresholds');
legend('EEG Ratio', 'Right Threshold', 'Left Threshold');
grid on;
hold off;