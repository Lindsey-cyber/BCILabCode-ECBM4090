% Lab2:
% Write scripts to:
% Plot raw EEG signals.
% Remove the DC component from each channel.
% Design filters for Alpha (8-13Hz) and Beta (14-32Hz) frequencies. Report power levels for each condition.

% Input:
% 30 seconds EEG of the 6 activities

% Data Analysis:
% 1. gbsanalyze: 
% preprocess: remove drift, delete sample count, set scale of 100 μV
% analyze: frequency distributions by Analyze, amplitude by Ruler
% 2. matlab analyze:

clear;
close all;

file_names = {
    '/Users/lindseyma/Desktop/EEG/data/Lab2/Lab2_Experiment1_Group5_Sitting_Quietly_1.mat',
    '/Users/lindseyma/Desktop/EEG/data/Lab2/Lab2_Experiment1_Group5_Blinking_1.mat',
    '/Users/lindseyma/Desktop/EEG/data/Lab2/Lab2_Experiment1_Group5_Rolling_Eyes_1.mat',
    '/Users/lindseyma/Desktop/EEG/data/Lab2/Lab2_Experiment1_Group5_Chewing_1.mat',
    '/Users/lindseyma/Desktop/EEG/data/Lab2/Lab2_Experiment1_Group5_Tensing_Neck_1.mat',
    '/Users/lindseyma/Desktop/EEG/data/Lab2/Lab2_Experiment1_Group5_Touching_Electrodes_1.mat'
   };

activity_names = {
    'Sitting Quietly',
    'Blinking', 
    'Rolling Eyes', 
    'Chewing', 
    'Tensing Neck', 
    'Touching Electrodes'
}

fs = 256;
time_duration = 30;

alpha_power = zeros(6,4);
beta_power = zeros(6,4);

for i = 1:length(file_names)
    data = load(file_names{i});
    eeg_data = data.y;
    t = (0:size(eeg_data,2) - 1) / fs;
    figure('Name',activity_names{i},'NumberTitle','off');
    for ch = 2:5
        subplot(4,1,ch-1);
        plot(t,eeg_data(ch, :));
        title([activity_names{i} ' - Channel ' num2str(ch)]);
        xlabel('Time(s)');
        ylabel('Amplitude(µV)');
        xlim([0 30]);
    end
    sgtitle(['EEG Data for ' activity_names{i}]);

    eeg_alpha = eegfilt(eeg_data(2:5,:), fs, 8, 13);
    eeg_beta = eegfilt(eeg_data(2:5,:), fs, 14, 32);
    alpha_power(i, :) = sum(eeg_alpha.^2, 2);
    beta_power(i, :) = sum(eeg_beta.^2, 2);
end

figure('Name', 'Alpha and Beta Power', 'NumberTitle', 'off');
for ch = 2:5
    subplot(4, 1, ch-1);
    bar([alpha_power(:,ch-1),beta_power(:,ch-1)]);
    title(['Channel ' num2str(ch)]);
    xlabel('Activity');
    ylabel('Power');
    xticks(1:length(activity_names));
    xticklabels(activity_names);
    legend({'Alpha Power', 'Beta Power'});
    xtickangle(45);
end
sgtitle('Alpha and Beta Power Across Different Activities');


% takeaway: Use bipolar derivation when the goal is to reduce interference
%from EOG and EMG artifacts, and Monopolar derivation may be more
%appropriate when a wider scope of brain activity is needed.