%% Prepara o ambiente
clc
clear all
%% Constantes
%Construtivas
F = 1;
V = 1;
Cae = 2.0;
Te = 323;
cp = 1;
rho = 10^6;
k0 = 10^10;
E_R = 8330.1;
lambda = 130*10^6;
Ta = 365;
%Aproximação do valor de UA
a = 1.678*10^6;
b = 0.5;
Fcs = 15;
Cpc = 1;
UA = (a*(Fcs)^(b+1))/(Fcs + a*(Fcs)^b/(2*rho*Cpc));
%Ponto de operação
Ts = 394;
Cs = 0.265;
%% Modelo linear para a concentração
a11 = F/V;
a12 = -(F/V + 2*k0*exp(-E_R/Ts));
a13 = (Cae-Cs)/V;
a14 = -(2*E_R*k0*exp(-E_R/Ts))*Cs/(Ts^2);
%% Modelo Linear para a temperatura
a21 = F/V;
a22 = lambda*k0*exp(-E_R/Ts)/(rho*cp);
a23 = UA/(rho*cp*V);
a24 = (-F/V+lambda*k0*E_R*exp(-E_R/Ts)/(rho*cp*Ts^2)*Cs-UA/(rho*cp*V));
%% Modelo em espaço de estados para o sistema linearizado
A = [a12 a14;a22 a24];
B = [a11 a13 0 0;0 0 a21 a23];
C = [1 0;0 1];
D = [0 0 0 0;0 0 0 0];
sys = ss(A,B,C,D);

%% Sim. 1:

dt = 0.1;
tfinal = 60;

x = zeros( tfinal /dt ,2) ; % Vetor de saidas
%u = [ zeros(10/ dt ,1); ones(50/ dt ,1) ]; % Vetor de entrada

%u = [ 0.2*randn(5*12/dt,1)]; %ruido branco 

u = [ ones(1,1) ; zeros((5/dt)-1 ,1)
      ones(1,1) ; zeros((5/dt)-1 ,1)
      ones(1,1) ; zeros((5/dt)-1 ,1)
      ones(1,1) ; zeros((5/dt)-1 ,1)
      ones(1,1) ; zeros((5/dt)-1 ,1)
      ones(1,1) ; zeros((5/dt)-1 ,1)
      ones(1,1) ; zeros((5/dt)-1 ,1)
      ones(1,1) ; zeros((5/dt)-1 ,1)
      ones(1,1) ; zeros((5/dt)-1 ,1)
      ones(1,1) ; zeros((5/dt)-1 ,1)
      ones(1,1) ; zeros((5/dt)-1 ,1)
      ones(1,1) ; zeros((5/dt)-1 ,1)]; %impulse





for i=2:tfinal/dt
[tout,xout] = ode45(@(t,x) SimulaTanque(t,x, [0 0 0 u(i)],A,B),[0 dt],x(i-1 ,:)');
x(i ,:) = xout (end ,:);
end

x = x + [Cs Ts];%Soma do ponto de operação ao step de entrada

subplot (3 ,1 ,1)
plot (0: dt: tfinal -dt ,x(:,1))
xlabel ( ' Tempo [s] ' )
ylabel ( ' Concentração ' )
legend ({ '' } )
subplot (3 ,1 ,2)
plot (0: dt: tfinal -dt ,x(:,2))
xlabel ( ' Temperatura [°C] ' )
ylabel ( ' Forca [N] ' )
legend ({ '' })
subplot (3 ,1 ,3)
plot (0: dt: tfinal -dt ,u)
xlabel ( ' Tempo [s] ' )
ylabel ( ' Concentração de entrada ' )
legend ({ '' })

save("Ta_var","x","u",'-mat');

%"Ca_var" => [u(i) 0 0 0] Step de Ca
%"Q_var" => [0 u(i) 0 0] Step de Q
%"Te_var" => [0 0 u(i) 0] Step de Te
%"Ta_var" => [0 0 0 u(i)] Stemp de Ta
% *%













function dx = SimulaTanque(t,x,u,A,B)
dx = zeros (2 ,1);
dx = A*x+B*u';
end






