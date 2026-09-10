clear; clc; close all;
x = 0.1:1/22:1;
target = (1 + 0.6*sin(2*pi*x/0.7) + 0.3*sin(2*pi*x)) / 2;
epoch = 10000;
eta = 0.1;

% Hidden layer.
w11_1 = rand(1);
b1_1 = 0;
w21_1 = rand(1);
b2_1 = 0;
% Output layer.
w11_2 = rand(1);
w21_2 = rand(1);
b1_2 = 0;
for j = 1:epoch
    for i = 1:length(x)
        % Forward pass.
        v1_1 = x(i)*w11_1 + b1_1;
        y1_1 = tanh(v1_1);
        v2_1 = x(i)*w21_1 + b2_1;
        y2_1 = tanh(v2_1);
        v1_2 = y1_1*w11_2 + y2_1*w21_2 + b1_2;
        y1_2 = v1_2;
        % Output error.
        e = target(i) - y1_2;

        % Calculate all deltas before updating weights.
        delta1_2 = e;
        delta1_1 = (1 - y1_1^2)*delta1_2*w11_2;
        delta2_1 = (1 - y2_1^2)*delta1_2*w21_2;

        % Update output weights and bias.
        w11_2 = w11_2 + eta*delta1_2*y1_1;
        w21_2 = w21_2 + eta*delta1_2*y2_1;
        b1_2 = b1_2 + eta*delta1_2;

        % Update hidden weights and biases.
        w11_1 = w11_1 + eta*delta1_1*x(i);
        b1_1 = b1_1 + eta*delta1_1;
        w21_1 = w21_1 + eta*delta2_1*x(i);
        b2_1 = b2_1 + eta*delta2_1;
    end

end

% Test after training.
x_naujas = 0.1:1/22:1;
target_naujas = (1 + 0.6*sin(2*pi*x_naujas/0.7) ...
    + 0.3*sin(2*pi*x_naujas)) / 2;
Y = zeros(size(x_naujas));
for i = 1:length(x_naujas)
    v1_1 = x_naujas(i)*w11_1 + b1_1;
    y1_1 = tanh(v1_1);
    v2_1 = x_naujas(i)*w21_1 + b2_1;
    y2_1 = tanh(v2_1);
    v1_2 = y1_1*w11_2 + y2_1*w21_2 + b1_2;
    y1_2 = v1_2;
    Y(i) = y1_2;
end
plot(x_naujas, target_naujas, '-o')
hold on
plot(x_naujas, Y, '-*')
legend('Target', 'Network')
