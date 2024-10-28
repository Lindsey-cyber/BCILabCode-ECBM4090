  
   t = (-30:30)';     
   Fc = 3;                     
   phi = 4 * pi / 3;
   x1 = sin(2*pi*Fc*t / 60 + phi);
   x2 = exp(1i*2*pi*Fc*t / 60);
   x3 = sqrt(3)/2 * real(x2) + 1/2 * imag(x2);
   figure;
   subplot(2,3,1)
   stem(t,x1)
   subplot(2,3,2)
   stem(t,real(x2))
   subplot(2,3,3)
   stem(t,imag(x2))
   subplot(2,3,4)
   stem(t,abs(x2))
   subplot(2,3,5)
   stem(t, angle(x2))
   subplot(2,3,6)
   stem(t,x3,".")
   xlabel('time (in seconds)');
   title('Signal versus Time');
   zoom xon;
   
   m = (-10:10)';
   theta = 2;
   h1 = zeros(size(m));
   h1(abs(m)>=2 & abs(m)<=6) = 1;
   h2 = zeros(size(m));
   h2(m == 1) = 1;
   h2(m == -1) = -1;
   h3 = zeros(size(m));
   h3 = exp(-m.^2 / (2*(theta^2)));
   y11 = conv(x1,h1);
   y12 = conv(x1,h2);
   y13 = conv(x1,h3);
   figure;
   subplot(2,3,1)
   stem(m,h1)
   subplot(2,3,2)
   stem(m,h2)
   subplot(2,3,3)
   stem(m,h3)
   subplot(2,3,4)
   stem(y11)
   subplot(2,3,5)
   stem(y12)
   subplot(2,3,6)
   stem(y13)