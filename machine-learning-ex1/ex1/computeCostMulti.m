function J = computeCostMulti(X, y, theta)

%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.


sum = 0;

for i = 1 : m,
    xi = X([i], :);                 % get the ith set of training data, which is the ith row from the matrix X
    x = xi';                        % get the transpose of xi for θTx
    hthetai = theta' * x;           % Compute the hθ(x)=θTx for the ith training set
    temp = hthetai - y([i],:);      % get the actual result for the ith training set
    sum = sum + (temp*temp);        % the sum of squares
end

J = sum / (2 * m);



% =========================================================================

end
