function [x,y,r] = EnclosedCurve(r_random,N_angle,H,Center_X,Center_Y)
ang=linspace(0,2*pi/N_angle*(N_angle-1),N_angle);

rho = rand(1,H)*0.15 .* logspace(-1.5,-2.5,H);
phi = rand(1,H) .* 2*pi;


r= zeros(1,N_angle)+r_random;

%temp =zeros(1,N_angle);
for h=1:H
   
  r = r + rho(h)*sin(h*ang+phi(h));
  %temp = temp+  rho(h)*sin(h*ang+phi(h));
%   for k = 1:N_angle
%       if r(k)>Rmax
%           r(k)=Rmax-0.01;
%       elseif r(k)<Rmin
%           r(k)=Rmin;
%       end
%   end
end
% rmin=min(r,[],'all');
% tmpmax = max(temp);
% tmpmin = min(temp);
x = r .* cos(ang)+Center_X;
y = r .* sin(ang)+Center_Y;
end

