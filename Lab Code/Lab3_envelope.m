function [envelope] = envelope_detection(raw_EEG_signal, band_low_cut_off, band_high_cut_off, low_cut_off)

    fs = 256;  
    filtered_signal = bandpass(raw_EEG_signal, [band_low_cut_off, band_high_cut_off], fs);

    envelope = lowpass(abs(filtered_signal), low_cut_off, fs);

end

raw_EEG_signal = EEGData0(2,:);
simulink_output = EEGData1(2,:) / 20000;

band_low_cut_off = 8;  
band_high_cut_off = 12; 
low_cut_off = 2;      

matlab_envelope = envelope_detection(raw_EEG_signal, band_low_cut_off, band_high_cut_off, low_cut_off);

time = EEGData1(1, :);

time_window_similar = [5120:5632];  
time_window_different = [2560:3072];  

figure;
plot(time(time_window_similar), matlab_envelope(time_window_similar), 'r-', 'DisplayName', 'MATLAB Envelope');
hold on;
plot(time(time_window_similar), simulink_output(time_window_similar), 'b-', 'DisplayName', 'Simulink Output');
title('Similar Outputs (MATLAB vs Simulink)');
xlabel('Time (s)');
ylabel('Amplitude');
legend;
hold off;

figure;
plot(time(time_window_different), matlab_envelope(time_window_different), 'r-', 'DisplayName', 'MATLAB Envelope');
hold on;
plot(time(time_window_different), simulink_output(time_window_different), 'b-', 'DisplayName', 'Simulink Output');
title('Different Outputs (MATLAB vs Simulink)');
xlabel('Time (s)');
ylabel('Amplitude');
legend;
hold off;
