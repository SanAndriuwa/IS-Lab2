% Step-by-step backpropagation example.
% This file shows ONE training step with two hidden neurons.
% The main Lab2 file uses the same formulas with eight neurons and 20 points.
clear; clc; close all;

% One training example.
x = 0.5;                 % input
d = 0.7;                 % desired output (target)
eta = 0.1;               % learning rate
H = 2;                   % use two neurons so the calculations are visible

% Fixed values make the first calculation repeatable.
w1 = [0.4; -0.7];        % input -> hidden weights (2x1)
b1 = [0.1; 0.2];         % hidden biases (2x1)
w2 = [0.6, -0.3];        % hidden -> output weights (1x2)
b2 = 0.1;                % output bias (scalar)

%% 1. Forward pass: calculate the hidden neurons.
z1 = w1*x + b1;          % weighted sums in the hidden layer
h = tanh(z1);             % hidden outputs after activation

%% 2. Forward pass: calculate the output neuron.
y = w2*h + b2;            % linear output
e = d - y;                % target minus current output

fprintf('Before update: y = %.4f, error = %.4f\n', y, e);

%% 3. Backpropagation: send the error to the hidden layer.
% The output is linear, so its derivative is 1.
delta2 = e;
% The derivative of tanh(z1) is 1 - h.^2.
delta1 = (w2' * delta2) .* (1 - h.^2);

%% 4. Update the output-layer parameters.
w2 = w2 + eta*delta2*h';
b2 = b2 + eta*delta2;

%% 5. Update the hidden-layer parameters.
w1 = w1 + eta*delta1*x;
b1 = b1 + eta*delta1;

fprintf('After update:  w2 = [%.4f %.4f], b2 = %.4f\n', w2, b2);
fprintf('Hidden weights: w1 = [%.4f %.4f]\n', w1(1), w1(2));

%% 6. Check the same example again.
hAfter = tanh(w1*x + b1);
yAfter = w2*hAfter + b2;
fprintf('After update:  y = %.4f, error = %.4f\n', ...
    yAfter, d-yAfter);

% The error will not always decrease after one step. Training uses
% many examples and many epochs so that the average error decreases.
