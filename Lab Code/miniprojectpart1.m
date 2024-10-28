clear; close all;
load('/Users/lindseyma/Downloads/MiniProject1_GroupData_Section2_2024/train1_group5_section2.mat');

signals = y(2:9,:);

onsets = y(10,:);
targets = y(11,:);

time = length(y);
dt = 1/256;

onset_indices = find(onsets > 0);
target_indices = find(targets > 0);

onset_window = - 26 * dt: dt: 130 * dt;
target_window = - 26 * dt: dt: 130 * dt;

average_onset = zeros(8, length(onset_window));
average_target = zeros(8, length(target_window));

for i = 1:8
    signal_onset = signals(i, onset_indices-26: onset_indices+130);
    signal_target = signals(i, target_indices-26: target_indices+130);

    average_onset(i,:) = mean(signal_onset, 2);
    average_onset(i,:) = mean(signal_target, 2);
end

time_vector = onset_window; 

figure;
for i = 1:8
    subplot(2, 1, 1); 
    plot(time_vector, average_onset(i, :));
    hold on; 
end
title('Average Onset Signals for Each Channel');
xlabel('Time (s)');
ylabel('Amplitude (\muV)');
legend({'Channel 1', 'Channel 2', 'Channel 3', 'Channel 4', 'Channel 5', 'Channel 6', 'Channel 7', 'Channel 8'}, 'Location', 'Best');
hold off;

for i = 1:8
    subplot(2, 1, 2); 
    plot(time_vector, average_target(i, :));
    hold on; 
end
title('Average Target Signals for Each Channel');
xlabel('Time (s)');
ylabel('Amplitude (\muV)');
legend({'Channel 1', 'Channel 2', 'Channel 3', 'Channel 4', 'Channel 5', 'Channel 6', 'Channel 7', 'Channel 8'}, 'Location', 'Best');
hold off;
