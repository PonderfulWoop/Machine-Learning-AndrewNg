function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

X = [ones(m, 1) X];

z2 = Theta1*X';
a2 = sigmoid(z2);
a2 = a2';

a2 = [ones(size(a2, 1), 1) a2];

z3 = Theta2*a2';
a3 = sigmoid(z3);
a3 = a3';

tempY = zeros(num_labels, m);
for u = 1:m
	tempY(y(u), u) = 1;
end
for i = 1:m
	for k = 1:num_labels
		J = J + (-1*tempY(k, i))*log(a3(i, k)) - (1 + -1*tempY(k, i))*log(1 - a3(i, k));
	end
end

J = (1/m)*J;

tempTheta1 = Theta1(:, 2:end);
tempTheta2 = Theta2(:, 2:end);

reg_add = sum(tempTheta1(:).^2) + sum(tempTheta2(:).^2);
reg_add = (lambda/(2*m))*reg_add;

J = J + reg_add;
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%del_3 = zeros(num_labels, 1);
DELTA_2 = zeros(size(Theta2));
DELTA_1 = zeros(size(Theta1));
for t = 1:m
	a_1 = X(t, :); % 1x401
	z_2 = Theta1*a_1'; % 25x1
	a_2 = sigmoid(z_2); % 25x1
	
	a_2 = [1; a_2]; % 26x1
	
	z_3 = Theta2*a_2; % 10x1
	a_3 = sigmoid(z_3); % 10x1
	
	del_3 = a_3 - tempY(:, t); % 10x1
	del_2 = Theta2'*del_3.*a_2.*(1-a_2); % 26x1
	del_2 = del_2(2:end); % 25x1
	
	DELTA_1 = DELTA_1 + del_2*a_1;
	DELTA_2 = DELTA_2 + del_3*a_2';
	
end
Theta1_grad = (1/m)*DELTA_1;
Theta2_grad = (1/m)*DELTA_2;
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
tempTheta1 = [zeros(size(Theta1, 1), 1) tempTheta1];
tempTheta2 = [zeros(size(Theta2, 1), 1) tempTheta2];

Theta1_grad = Theta1_grad + (lambda/m)*tempTheta1;
Theta2_grad = Theta2_grad + (lambda/m)*tempTheta2;

















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
