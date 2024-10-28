f = 440;            
duration = 5;       
fs = 44100;         

t = 0:1/fs:duration;

W = sin(2*pi*f*t);

amplitude = sin(2*pi*t/duration); 
W_modulated = amplitude .* W;   

soundsc(W_modulated, fs);

figure;
plot(t(1:500), W_modulated(1:500));  
title('Amplitude Modulated Sine Wave');
xlabel('Time (s)');
ylabel('Amplitude');
