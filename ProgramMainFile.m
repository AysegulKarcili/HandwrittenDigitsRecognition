clear ; close all;
% Handwritten digits recognition

% Load Data
load('data.mat'); % X and y

% Preparation for Neural Network
m = size(X, 1); % number of training examples (equal 5000)
input_layer_size  = size(X, 2);  % 20x20 Input Images of Digits (equals 400)
hidden_layer_size = 25;   % 25 hidden units
num_labels = length(unique(y)); % number of labels, from 1 to 10 ("0" is mapped to label 10)
lambda = 1; % 3 % regularization parameter
num_iters = 100; % 50 % number of iterations

% Display randomly selected 100 data points i.e. handwritten digit images 
sel = randperm(m);
sel = sel(1:100);
figure;
displayData(X(sel, :));

% Initialize Neural Network parameters and unroll them 
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

% Compute regularized cost by feedforwarding and compute partial
% derivatives of the cost function w.r.t. parameters by backpropagation
[J, gradVec] = nnCostFunction(initial_nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);
fprintf('Cost at initial parameters with regularization parameter lambda = %f : %f \n', lambda, J);

% Create short hand for the cost function to be minimized
costFunc = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);
% costFunc is a function that takes in only one argument, the neural network parameters

% Train the neural network
options = optimset('MaxIter', num_iters);
[nn_params, cost] = fmincg(costFunc, initial_nn_params, options); % here it is possible to use 

% Reroll Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1 : hidden_layer_size*(1+input_layer_size)), hidden_layer_size, 1+input_layer_size);
Theta2 = reshape(nn_params(hidden_layer_size*(1+input_layer_size) + 1 : end), num_labels, 1+hidden_layer_size);

% Visualizations:
% We have features from input layer and we have displayed 100 random
% examples among them. 
% We get Theta1 by training the neural network and we can visualize it to
% see which weights it is giving to the features of the data to capture
% info. 
% We can get hidden layer units and we can display 100 random examples
% among them to see what features it has constracted from the data. 
% We also get Theta2 and we can visualize it to see which weights it is
% giving to the artificial features of the hidden layer to capture ultimate
% info. 
% We can also get output layer units and we can display 100 random examples
% among them to visualize what labels it has produced from the data. 
% To make all the visualizations we are artifically rerolling the rows of
% the matrices. sqrt(400), sqrt(25), sqrt(25), sqrt(10-1) discard
% propabilities of being 0 for just integer squareroot purpose to
% visualize.

%------------
figure;
displayData(Theta1(:, 2:end)); % size: hidden_layer_size x (1+input_layer_size)

a1 = [ones(m,1) X]; % size: m x (1+input_layer_size)
a2 = sigmoid(a1*Theta1'); % size: m x hidden_layer_size
figure;
displayData(a2(sel, :));

figure;
displayData(Theta2(:, 2:end)); % size: num_labels x (1+hidden_layer_size)

a2 = [ones(size(a2,1),1) a2]; % size: m x (1+hidden_layer_size)
a3 = sigmoid(a2*Theta2'); % size: m x num_labels
figure;
displayData(a3(sel, 1:end-1));
%------------

% Make predictions on the training set using the N.N. and calculate accuracy 
pred = predict(X, Theta1, Theta2);
accuracy = mean(double(pred == y)) * 100;
fprintf('Training Set Accuracy: %f\n', accuracy);
