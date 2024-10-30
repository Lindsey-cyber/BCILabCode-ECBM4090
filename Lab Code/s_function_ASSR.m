function s_function_ASSR (block)
% Level-2 MATLAB file S-Function for times two demo.
%   Copyright 1990-2009 The MathWorks, Inc.
%   $Revision: 1.1.6.2 $ 

setup(block);
function setup(block)  
  %% Register number of input and output ports
  block.NumInputPorts  = 1;
  block.NumOutputPorts = 1;

  %% Setup functional port properties to dynamically
  %% inherited.
  block.SetPreCompInpPortInfoToDynamic;
  block.SetPreCompOutPortInfoToDynamic;
  block.InputPort(1).DirectFeedthrough = true;  
  %% Set block sample time to inherited
  block.SampleTimes = [-1 0];
  
  %% Set the block simStateCompliance to default (i.e., same as a built-in block)
  block.SimStateCompliance = 'DefaultSimState';
 
  %% Register methods
  block.RegBlockMethod('Start',@Start);
  block.RegBlockMethod('Outputs',                 @Output);  
end

function Start(block)
  disp('Running...')
  CurrentTime = 0;
  global ct;
  global w;

end


fs = 44100;
fc = 900;
fm = 40;
d = 10; % or 6000s
t = (1 : d * fs) / fs;
wc = sin(2 * pi * fc * t);
wm = 1 + sin(2 * pi * fm * t);

%flag_onset = zeros(size(t));
%flag_onset(t >= delay) = 1;
%pause(delay);
%soundsc(w);

function Output(block)
    
  w = wc .* wm;
  ct = get(block, 'CurrentTime');

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
 
  if ct == 15
      block.OutputPort(1).Data(1) = double(1);
      soundsc(w, fs);
  end

  if ct == 615
      block.OutputPort(1).Data(1) = double(0);
      clear sound;
  end

end
end