function predictions = predict(X, Theta1, Theta2)
% Predicts the label of an input X given a trained neural network (Theta1,
% Theta2) 

m = size(X, 1); % number of training examples

a2 = sigmoid([ones(m, 1) X] * Theta1');
a3 = sigmoid([ones(m, 1) a2] * Theta2');
[~, predictions] = max(a3, [], 2);

end