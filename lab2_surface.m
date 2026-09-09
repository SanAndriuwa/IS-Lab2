% Additional task: two inputs and one output, surface approximation.
clear; clc; close all;
rng(2);

% The assignment does not specify a surface, so we choose a simple smooth one.
[X1, X2] = meshgrid(linspace(0, 1, 10));
P = [X1(:)'; X2(:)'];
d = 0.5 + 0.25*sin(pi*P(1, :)).*cos(pi*P(2, :));
H = 8;
W1 = randn(H, 2);
b1 = randn(H, 1);
w2 = 0.3*randn(1, H);
b2 = 0;
eta = 0.02;
maxEpochs = 10000;
loss = zeros(1, maxEpochs);

for epoch = 1:maxEpochs
    for i = 1:size(P, 2)
        h = tanh(W1*P(:, i) + b1);
        y = w2*h + b2;
        e = d(i) - y;
        delta = (w2'*e) .* (1 - h.^2);
        w2 = w2 + eta*e*h';
        b2 = b2 + eta*e;
        W1 = W1 + eta*delta*P(:, i)';
        b1 = b1 + eta*delta;
    end
    yTrain = zeros(size(d));
    for i = 1:size(P, 2)
        yTrain(i) = w2*tanh(W1*P(:, i) + b1) + b2;
    end
    loss(epoch) = mean((d - yTrain).^2);
    if loss(epoch) < 1e-4
        break;
    end
end
loss = loss(1:epoch);

[A, B] = meshgrid(linspace(0, 1, 41));
Z = 0.5 + 0.25*sin(pi*A).*cos(pi*B);
Znet = zeros(size(Z));
for i = 1:numel(A)
    Znet(i) = w2*tanh(W1*[A(i); B(i)] + b1) + b2;
end
fprintf('Epochs: %d\n', epoch);
fprintf('Training MSE: %.6f\n', loss(end));
fprintf('Dense-grid MSE: %.6f\n', mean((Z(:)-Znet(:)).^2));
disp('Hidden-layer weights W1:'); disp(W1);
disp('Hidden-layer biases b1:'); disp(b1);
disp('Output weights w2:'); disp(w2);
fprintf('Output bias b2: %.6f\n', b2);

figure;
subplot(1, 2, 1); surf(A, B, Z); title('Target surface');
xlabel('x1'); ylabel('x2'); zlabel('y'); zlim([0.2 0.8]);
subplot(1, 2, 2); surf(A, B, Znet); title('Network surface');
xlabel('x1'); ylabel('x2'); zlabel('y'); zlim([0.2 0.8]);
figure;
semilogy(loss); xlabel('Epoch'); ylabel('MSE'); grid on;
title('Training error');

