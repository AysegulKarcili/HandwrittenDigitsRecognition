function [jVal, gradVec] = nnCostFunction(thetaVec, input_layer_size, hidden_layer_size, num_labels, X, y, lambda)

% Reroll the N.N. parameters
Theta1 = reshape(thetaVec(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(thetaVec((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));

m = length(y); % number of training examples

% Using the vector y, construct a binary matrix y_binary which includes 1's
% on the column of the given number and 0's otherwise.
y_binary = zeros(m, num_labels); 
for k = 1:num_labels
    y_binary(y == k, k) = 1;
end

% Feedforward the N.N. to calculate cost
a1 = X; % size: m x input_layer_size
a1 = [ones(m,1) a1]; % size: m x (1+input_layer_size)
a2 = sigmoid(a1*Theta1'); % size: m x hidden_layer_size
a2 = [ones(m,1) a2]; % size: m x (1+hidden_layer_size)
a3 = sigmoid(a2*Theta2'); % size: m x num_labels
% Note that a1 and a2 include 1's column but a3 doesn't for 3-layer network.

% Calculate the cost J 
jVal = (-1/m) * sum(sum( y_binary.*log(a3) + (1-y_binary).*log(1-a3) )) + ... 
    (lambda/(2*m)) * (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2))); % exclude the leftmost columns of thetas in regularization 

% Backpropogate the N.N. to calculate gradients
D2 = zeros(size(Theta2));
D1 = zeros(size(Theta1));
for t = 1:m
    delta3 = a3(t,:)' - y_binary(t,:)';
    delta2 = Theta2' * delta3 .* a2(t,:)' .* (1-a2(t,:)');
    delta2 = delta2(2:end); % exclude the uppermost value corresponding to a0=1 of layer l. 
    
    D2 = D2 + delta3 * a2(t,:);
    D1 = D1 + delta2 * a1(t,:);
end

% Replace the leftmost column of theta with 0's to use in regularization
Theta1_temp = [zeros(size(Theta1,1),1) Theta1(:, 2:end)]; 
Theta2_temp = [zeros(size(Theta2,1),1) Theta2(:, 2:end)];

% Average and regularize the gradients
D1 = (1/m)*D1 + (lambda/m)*Theta1_temp; 
D2 = (1/m)*D2 + (lambda/m)*Theta2_temp;

% Unroll the gradients
gradVec = [D1(:) ; D2(:)];

end