% Основное задание: сеть с одним входом, 8 скрытыми нейронами и выходом.
% Скрытая активация tanh, выходная активация линейная.
clear; clc; close all;
rng(1); % одинаковый результат при повторном запуске

x = linspace(0, 1, 20);
% В README лишняя закрывающая скобка. Здесь исправлена группировка.
d = (1 + 0.6*sin(2*pi*x/0.7) + 0.3*sin(2*pi*x)) / 2;
H = 8;
w1 = randn(H, 1);
b1 = randn(H, 1);
w2 = 0.3*randn(1, H);
b2 = 0;
eta = 0.02;
maxEpochs = 100000;
loss = zeros(1, maxEpochs);

for epoch = 1:maxEpochs
    for i = 1:length(x)
        % Прямой проход: получаем ответ сети.
        h = tanh(w1*x(i) + b1);
        y = w2*h + b2;
        e = d(i) - y;

        % Обратное распространение ошибки.
        % Производная tanh равна 1 - h^2.
        delta = (w2' * e) .* (1 - h.^2);
        % delta вычисляется до изменения весов выходного слоя.
        w2 = w2 + eta*e*h';
        b2 = b2 + eta*e;
        w1 = w1 + eta*delta*x(i);
        b1 = b1 + eta*delta;
    end

    % Ошибка с текущими весами после полного прохода по данным.
    yTrain = zeros(size(x));
    for i = 1:length(x)
        yTrain(i) = w2*tanh(w1*x(i) + b1) + b2;
    end
    loss(epoch) = mean((d - yTrain).^2);
    if loss(epoch) < 1e-4
        break;
    end
end
loss = loss(1:epoch);

% Плотная сетка показывает поведение между обучающими точками.
xTest = linspace(0, 1, 201);
dTest = (1 + 0.6*sin(2*pi*xTest/0.7) + 0.3*sin(2*pi*xTest)) / 2;
yTest = zeros(size(xTest));
for i = 1:length(xTest)
    yTest(i) = w2*tanh(w1*xTest(i) + b1) + b2;
end
fprintf('Эпох: %d\n', epoch);
fprintf('MSE на обучении: %.6f\n', loss(end));
fprintf('MSE на плотной сетке: %.6f\n', mean((dTest-yTest).^2));
disp('Коэффициенты скрытого слоя:');
disp(table(w1, b1));
disp('Веса выхода w2:'); disp(w2);
fprintf('Смещение выхода b2: %.6f\n', b2);

figure;
plot(xTest, dTest, 'b-', xTest, yTest, 'r--', x, d, 'ko', 'LineWidth', 1.5);
legend('Target function', 'Network', 'Training points', 'Location', 'best');
xlabel('x'); ylabel('y'); grid on; title('Function approximation');
figure;
semilogy(loss); xlabel('Epoch'); ylabel('MSE'); grid on;
title('Training error');
