function [x_test, target_test, Y_test, mse_test] = test_network(w1, b1, w2, b2, x_start, x_step, x_end)
% TEST_NETWORK Проверить обученную сеть на другой сетке без обучения.
% Пример: test_network([w11_1;w21_1], [b1_1;b2_1], ...
%                     [w11_2,w21_2], b1_2, 0.05, 1/22, 1);
% w1, b1: Hx1; w2: 1xH; b2: число. Подходит для H=2 и H=4.
% Последняя точка <= x_end; оператор : не обязан включать x_end.
if nargin < 5, x_start = 0.05; end
if nargin < 6, x_step = 1/22; end
if nargin < 7, x_end = 1; end
validateattributes(w1, {'numeric'}, {'real','finite','vector','nonempty'});
validateattributes(b1, {'numeric'}, {'real','finite','vector','numel',numel(w1)});
validateattributes(w2, {'numeric'}, {'real','finite','vector','numel',numel(w1)});
validateattributes(b2, {'numeric'}, {'real','finite','scalar'});
validateattributes(x_start, {'numeric'}, {'real','finite','scalar'});
validateattributes(x_step, {'numeric'}, {'real','finite','scalar','positive'});
validateattributes(x_end, {'numeric'}, {'real','finite','scalar','>=',x_start});
w1 = w1(:); b1 = b1(:); w2 = w2(:)';
x_test = x_start:x_step:x_end;
target_test = (1 + 0.6*sin(2*pi*x_test/0.7) + 0.3*sin(2*pi*x_test)) / 2;
Y_test = zeros(size(x_test));
for i = 1:length(x_test)
    v_hidden = w1*x_test(i) + b1;
    y_hidden = tanh(v_hidden);
    Y_test(i) = w2*y_hidden + b2;
end
mse_test = mean((target_test - Y_test).^2);
fprintf('Test grid: start=%.5f, step=%.8f, last=%.8f, N=%d\n', ...
    x_start, x_step, x_test(end), numel(x_test));
fprintf('Test-grid MSE: %.8f\n', mse_test);
figure;
plot(x_test, target_test, 'bo-', x_test, Y_test, 'r*-');
legend('Target function', 'Trained network', 'Location', 'best');
xlabel('x'); ylabel('y'); grid on;
title('Test on a shifted input grid (no training)');
end
