clc
clear
close all

hold on


x=0.1:1/22:1;

target = (1 + 0.6 * sin (2 * pi * x / 0.7)) + 0.3 * sin (2 * pi * x) / 2;

plot(x,target,'-o')

epoch=10000;
eta=0.1;

%pirmas sl.
w11_1=rand(1);
w21_1=rand(1);
b1_1=0;
b2_1=b1_1;

%antras sl.
w11_2=rand(1);
w21_2=rand(1);
b1_2=0;


%feedworard

for j=1:epoch
    for i=1:length(x)
        v1_1=x(i)*w11_1+b1_1;
        v2_1=x(i)*w21_1+b2_1;
    
        %aktyvacijos funkcija
        y1_1=tanh(v1_1);
        y2_1=tanh(v1_1);
    
        %2sl. weighted summ
        v1_2=y1_1 * w11_2 + y2_1 + w11_2 + b1_2;
    
        %isejino sluoknsio aktyvacija
        y1_2=v1_2;
    
        %tinklo isejimas
        target=y1_2;
    
        %klaida
        error=target(1)-target;
    
        %svorio atnaujinimas
        %isejimo
        delta1_2=error;
        %hidden
        delta1_1=(1-tanh(v1_1)^2)*delta1_2*w11_2;
        delta2_1=(1-tanh(v1_1)^2)*delta1_2*w21_2;
    
        %weight update
        %isejimo sluoksniai
        w11_2=w11_2*eta*delta1_2*y1_1;
        w12_2=w21_2*eta*delta2_1*y1_1;
        b1_2=b1_2*eta*delta1_2;
    
        %hidden layer svoriai
        w11_2=w11_2*eta*delta1_1*(x(i));
        w12_2=w12_2*eta*delta2_1*(x(i));
        b1_1=b1_1*eta*delta1_1;
        b1_2=b2_1*eta*delta2_1;
    end
end

x_naujas=0.1:1/22:1;
Y=zeros(1,length(x_naujas));

for i=1:length(x)
    v1_1=x(i)*w11_1+b1_1;
    v2_1=x(i)*w21_1+b2_1;

    %aktyvacijos funkcija
    y1_1=tanh(v1_1);
    y2_1=tanh(v1_1);

    %2sl. weighted summ
    v1_2=y1_1 * w11_2 + y2_1 + w12_2 + b1_2;

    %isejino sluoknsio aktyvacija
    y1_2=v1_2;

    %tinklo isejimas
    Y(i)= y1_2;
end

plot(x,Y,'-*')

