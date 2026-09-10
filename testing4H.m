clear; 
clc; 
close all;
x = 0.1:1/22:1;
target = (1 + 0.6*sin(2*pi*x/0.7) + 0.3*sin(2*pi*x)) / 2;
epoch = 100000;
eta = 0.1;

%% Hidden 1-2
w11_1 = rand(1);
b1_1 = 0;
w21_1 = rand(1);
b2_1 = 0;
% Hidden 3-4
w31_1 = rand(1);
b3_1 = 0;
w41_1 = rand(1);
b4_1 = 0;

%% Output 1-2
w11_2 = rand(1);
w21_2 = rand(1);
% Output 3-4
w31_2 = rand(1);
w41_2 = rand(1);

b1_2 = 0; %bendras outputas

y_mokymas=zeros(1,length(x));
%%
for j = 1:epoch
    for i = 1:length(x)
        %% Forward 1-2
        v1_1 = x(i)*w11_1 + b1_1;
        y1_1 = tanh(v1_1);
        v2_1 = x(i)*w21_1 + b2_1;
        y2_1 = tanh(v2_1);
        %Forawrd 2-4
        v3_1 = x(i)*w31_1 + b3_1;
        y3_1 = tanh(v3_1);
        v4_1 = x(i)*w41_1 + b4_1;
        y4_1 = tanh(v4_1);
        %%

        v1_2 = y1_1*w11_2 + y2_1*w21_2 + y3_1*w31_2 + y4_1*w41_2 + b1_2;

        y1_2 = v1_2;

        y_mokymas(i)= y1_2;
        %% Output
        e = target(i) - y1_2;

        %% Delta apskaiciavimas
        delta1_2 = e;
        delta1_1 = (1 - y1_1^2)*delta1_2*w11_2;
        delta2_1 = (1 - y2_1^2)*delta1_2*w21_2;
        delta3_1 = (1 - y3_1^2)*delta1_2*w31_2;
        delta4_1 = (1 - y4_1^2)*delta1_2*w41_2;

        %% svoriu atnaujinimas
        w11_2 = w11_2 + eta*delta1_2*y1_1;
        w21_2 = w21_2 + eta*delta1_2*y2_1;
        w31_2 = w31_2 + eta*delta1_2*y3_1;
        w41_2 = w41_2 + eta*delta1_2*y4_1;
        b1_2 = b1_2 + eta*delta1_2;

        %% paslepto sl. svoriu atnaujinimas
        w11_1 = w11_1 + eta*delta1_1*x(i);
        b1_1 = b1_1 + eta*delta1_1;
        w21_1 = w21_1 + eta*delta2_1*x(i);
        b2_1 = b2_1 + eta*delta2_1;
        w31_1 = w31_1 + eta*delta3_1*x(i);
        b3_1 = b3_1 + eta*delta3_1;
        w41_1 = w41_1 + eta*delta4_1*x(i);
        b4_1 = b4_1 + eta*delta2_1;
    end

end
fig1=figure;
fig1.Name = 'Apmokymas';
plot(x, target, '-o')
hold on
plot(x, y_mokymas, '-*')
legend('Target', 'Tinklas')


%% testavimas
x_naujas = 0.01:1/100:1;
target_naujas = (1 + 0.6*sin(2*pi*x_naujas/0.7) + 0.3*sin(2*pi*x_naujas)) / 2;
Y = zeros(size(x_naujas));
for i = 1:length(x_naujas)
    v1_1 = x_naujas(i)*w11_1 + b1_1;
    y1_1 = tanh(v1_1);
    v2_1 = x_naujas(i)*w21_1 + b2_1;
    y2_1 = tanh(v2_1);
    v3_1 = x_naujas(i)*w31_1 + b3_1;
    y3_1 = tanh(v3_1);
    v4_1 = x_naujas(i)*w41_1 + b4_1;
    y4_1 = tanh(v4_1);
    v1_2 = y1_1*w11_2 + y2_1*w21_2 + y3_1*w31_2 + y4_1*w41_2 + b1_2;
    y1_2 = v1_2;
    Y(i) = y1_2;
end

fig2=figure;
fig2.Name = 'Testavimas';
title('Testavimas');
plot(x_naujas, target_naujas, '-o')
hold on
plot(x_naujas, Y, '-*')
legend('Target', 'Tinklas')