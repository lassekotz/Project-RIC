%clear all; clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Fysikaliska parametrar för roboten. Ändra utifrån prototyp.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Balanduino.m  = 0.38;      % [kg]  massa for batteri och oversta hyllan med skruvar
Balanduino.M  = 0.97;      % [kg]  massan for resten av robotan
Balanduino.d  = 0.18;      % [m]   avstand mellan hjulaxel och batteriets masscentrum
Balanduino.r  = 0.049;     % [m]   hjulets radie
Balanduino.Ra = 2.4;       % [Ohm] inre resistansen for DC-motorn
Balanduino.La = 0.00025;   % [H]   inre induktansen for DC-motorn
Balanduino.Km = 0.155;     % [-]   omvandlingsfaktor mellan strom och moment 
Balanduino.Ku = 0.3078;    % [-]   omvandlingsfaktor for DC-motorns mot-EMK 
Balanduino.g  = 9.81;      % [m/s^2] 
% Max/min spanning till motorerna
Balanduino.u_max =  12; % [V]
Balanduino.u_min = -12; % [V]

% Samplingstid for den diskreta regulatorn
Balanduino.Ts = 0.004; % exekveringstid for programmet

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

g = Balanduino.g;
m = Balanduino.m;
M = Balanduino.M;
d = Balanduino.d;
r = Balanduino.r;
Ra = Balanduino.Ra;
La = Balanduino.La;
Km = Balanduino.Km;
Ku = Balanduino.Ku;
Ts = Balanduino.Ts;

A = [0 1 0;
    g/d*(1+m/M) 0 0;
    g*m/M 0 0];

B = [0 Km/(r*Ra*M*d) Km/(r*Ra*M)]';

C = [1 0 0;
     0 0 1];

D = 0;

sys = ss(A,B,C,D);

Q = diag([100,10,100]);
R = 1;

K = lqrd(A,B,Q,R,Ts);

Aaug = [A zeros(3,2);
        1 0 0 0 0;
        0 0 1 0 0];

Baug = [B;0;0];

Caug = [C zeros(2,2)];


Qaug = diag([100,10,10,100,10]);
R = 0.01;

%Kaug = lqrd(Aaug,Baug,Qaug,R,Ts)




