% Lab2: один вход, 2 скрытых нейрона tanh, один линейный выход.
% Отдельные переменные сохранены, как в исходном testing.m.
clear; clc; close all;
rng(1); % Повторяемые начальные веса.
x = 0.1:1/22:1;
target = (1 + 0.6*sin(2*pi*x/0.7) + 0.3*sin(2*pi*x)) / 2;
epoch = 10000;
eta = 0.1;

% Первый слой: у каждого нейрона свой входной вес и смещение.
w11_1 = rand(1);
b1_1 = 0;
w21_1 = rand(1);
b2_1 = 0;
% Второй слой: по одному весу от каждого скрытого нейрона.
w11_2 = rand(1);
w21_2 = rand(1);
b1_2 = 0;
Yinitial = zeros(size(x));
for i = 1:length(x)
    v1_1 = x(i)*w11_1 + b1_1;
    y1_1 = tanh(v1_1);
    v2_1 = x(i)*w21_1 + b2_1;
    y2_1 = tanh(v2_1);
    v1_2 = y1_1*w11_2 + y2_1*w21_2 + b1_2;
    y1_2 = v1_2; % Линейная активация выходного нейрона.
    Yinitial(i) = y1_2;
end
initialMSE = mean((target - Yinitial).^2);
loss = zeros(1, epoch);

for j = 1:epoch
    for i = 1:length(x)
        % 1. Прямой проход: взвешенные суммы, tanh, ответ сети.
        v1_1 = x(i)*w11_1 + b1_1;
        y1_1 = tanh(v1_1);
        v2_1 = x(i)*w21_1 + b2_1;
        y2_1 = tanh(v2_1);
        v1_2 = y1_1*w11_2 + y2_1*w21_2 + b1_2;
        y1_2 = v1_2; % Линейная активация выходного нейрона.
        % 2. Правильный ответ НЕ заменяем прогнозом сети.
        e = target(i) - y1_2;

        % 3. Все delta считаем ДО изменения любых весов.
        % Для E = 0.5*e^2 delta имеет знак минус градиента по v.
        delta1_2 = e;
        delta1_1 = (1 - y1_1^2)*delta1_2*w11_2;
        delta2_1 = (1 - y2_1^2)*delta1_2*w21_2;

        % 4. Выходной слой: старое значение ПЛЮС поправка.
        w11_2 = w11_2 + eta*delta1_2*y1_1;
        w21_2 = w21_2 + eta*delta1_2*y2_1;
        b1_2 = b1_2 + eta*delta1_2;

        % 5. Скрытый слой: обновляем его собственные веса и bias.
        w11_1 = w11_1 + eta*delta1_1*x(i);
        b1_1 = b1_1 + eta*delta1_1;
        w21_1 = w21_1 + eta*delta2_1*x(i);
        b2_1 = b2_1 + eta*delta2_1;
    end

    % MSE после эпохи: все точки проверяются с одними текущими весами.
    Ytrain = zeros(size(x));
    for i = 1:length(x)
        v1_1 = x(i)*w11_1 + b1_1;
        y1_1 = tanh(v1_1);
        v2_1 = x(i)*w21_1 + b2_1;
        y2_1 = tanh(v2_1);
        v1_2 = y1_1*w11_2 + y2_1*w21_2 + b1_2;
        y1_2 = v1_2; % Линейная активация выходного нейрона.
        Ytrain(i) = y1_2;
    end
    loss(j) = mean((target - Ytrain).^2);
end

% Более плотная сетка показывает поведение между учебными точками.
% Здесь нет обучения. Это проверка аппроксимации, не независимый test set.
x_naujas = linspace(0.1, 1, 201);
target_naujas = (1 + 0.6*sin(2*pi*x_naujas/0.7) ...
    + 0.3*sin(2*pi*x_naujas)) / 2;
Y = zeros(size(x_naujas));
for i = 1:length(x_naujas)
    v1_1 = x_naujas(i)*w11_1 + b1_1;
    y1_1 = tanh(v1_1);
    v2_1 = x_naujas(i)*w21_1 + b2_1;
    y2_1 = tanh(v2_1);
    v1_2 = y1_1*w11_2 + y2_1*w21_2 + b1_2;
    y1_2 = v1_2; % Линейная активация выходного нейрона.
    Y(i) = y1_2;
end
fprintf('Hidden neurons: 2\n');
fprintf('Initial training MSE: %.8f\n', initialMSE);
fprintf('Final training MSE: %.8f\n', loss(end));
fprintf('Dense-grid MSE: %.8f\n', mean((target_naujas - Y).^2));

figure;
plot(x, target, 'ko', x_naujas, target_naujas, 'b-', ...
    x_naujas, Y, 'r--', 'LineWidth', 1.5);
legend('Training points', 'Target function', 'Network', 'Location', 'best');
xlabel('x'); ylabel('y'); grid on;
title('2 hidden neurons: function approximation');
figure;
semilogy(1:epoch, loss);
xlabel('Epoch'); ylabel('Training MSE'); grid on;
title('Error after each epoch');

% Проверка на новых входах с тем же шагом; веса не меняются.
addpath(fileparts(mfilename('fullpath')));
[x_test, target_test, Y_test, mse_test] = test_network( ...
    [w11_1; w21_1], [b1_1; b2_1], ...
    [w11_2, w21_2], b1_2, 0.05, 1/22, 1);
