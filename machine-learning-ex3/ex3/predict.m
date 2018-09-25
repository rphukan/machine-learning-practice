function p = predict(Theta1, Theta2, X)

%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%


% for each of the m training sets
for training_set_count = 1 : m,

    a1 = X(training_set_count, :);
    % convert it to a vector
    a1 = a1';
    % Add ones as a0 to the a1 data matrix
    a1 = [1; a1];

    z2 = Theta1 * a1;
    a2 = sigmoid(z2);
    % Add ones as a0 to the a2 data matrix
    a2 = [1; a2];

    z3 = Theta2 * a2;
    a3 = sigmoid(z3);

    % a3 is a (10,1) vector containing the probabilities of the input being 1 to 10(0)
    [probability, index] = max(a3);  % the max probability, the index is the no

    p(training_set_count, 1) = index;

end




% =========================================================================


end
