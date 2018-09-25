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
%
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
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

DELTA1 = 0;
DELTA2 = 0;

%fprintf('\n Size of Theta1 : %f \n', size(Theta1));
%fprintf('\n Size of Theta2 : %f \n', size(Theta2));

% compute the cost without the regularization part
for training_set_count = 1 : m,

    % STEP 1 : use forward propagation and compute the h(x)
    xi = X(training_set_count, :);              %pick up a row , which is a training set
    a1 = xi';                                   % convert it to a vector
    a1 = [1; a1];                               % Add ones as a0 to the a1 data matrix

    z2 = Theta1 * a1;
    a2 = sigmoid(z2);
    a2 = [1; a2];                               % Add ones as a0 to the a2 data matrix

    z3 = Theta2 * a2;
    a3 = sigmoid(z3);                           % a3 is a (num_labels ,1) vector containing the probabilities of the input being 1 to K

    % Theta1 is (25, 401)
    % Theta2 is (10, 26)
    % a1 is (401,1)
    % z2 is (25, 1)
    % a2 is (26, 1)
    % z3 is (10, 1)
    % a3 is (10, 1)

    %STEP 2 : take yi and convert it to a vector of 0 and 1
    yi = y(training_set_count,1);
    yk = [1:num_labels]';                       % create a vector of 1 to K
    yk = (yk == yi);                            % make the yi element 1 and rest 0 s

    %STEP 3 : This vector multiplication would mean summation of K = 1 to K
    J = J + (  yk' * log(a3)  +  (1 - yk') * log(1 - a3)  );

    % yk is (10, 1)
    % yk' is (1, 10)
    % a3 is (10, 1)
    % hence J is a (1, 1) scalar value

    %For back propagation
    delta3 = a3 - yk;                                    %both are (num_labels, 1) vectors
    delta2 = (Theta2' * delta3) .* a2 .* (1 - a2);

    DELTA2 = DELTA2 + delta3 * a2';                     % delta3(10, 1) * a2'(1, 26) = DELTA2(10, 26)
    DELTA1 = DELTA1 + delta2(2:end) * a1';              % delta2(26, 1) , delta2(2:end) is (25, 1) * a1'(1, 401) = DELTA1(25, 401)

    % delta3 is (10, 1)
    % delta2 is (26, 1)
    % DELTA2 is (10, 26)
    % DELTA1 is (25, 401)

end

% compute the regularization cost

regTheta1 = Theta1(:, 2 : (input_layer_size + 1));      %remove the first Theta 0 column
regTheta1 = regTheta1(:);                               %create a vector from the Theta matrix

regTheta2 = Theta2(:, 2 : (hidden_layer_size + 1));
regTheta2 = regTheta2(:);

regularizedCost = regTheta1' * regTheta1 + regTheta2' * regTheta2;   %get the summation

J = - J / m + (lambda * regularizedCost) / (2 * m);

% compute the regularised part for gradient for J > 0
regGrad1 = Theta1(:,2:end) * lambda / m;     % regGrad1(25, 400) after removing the first row for j=0
regGrad2 = Theta2(:,2:end) * lambda / m;     % regGrad1(10, 25) after removing the first row for j=0

% calculate gradient
Theta1_grad = DELTA1/m;
Theta1_grad = [Theta1_grad(:, 1), Theta1_grad(:, 2:end) + regGrad1];   %keep the first column (j=0 )as it is and add regGrad1 to the remaining columns

Theta2_grad = DELTA2/m;
Theta2_grad = [Theta2_grad(:, 1), Theta2_grad(:, 2:end) + regGrad2];   %keep the first column (j=0) as it is and add regGrad2 to the remaining columns

% Theta1(25, 401) = Theta1_grad(25, 401) = DELTA1(25, 401)
% Theta2(10, 26) = Theta2_grad(10, 26) = DELTA2(10, 26)

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
