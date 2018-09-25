function [J, grad] = linearRegCostFunction(X, y, theta, lambda)

%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


sumJ = 0;
sumGrad = 0;

%for i = 1 : m,
%    xi = X([i], :);                 % get the ith set of training data, which is the ith row from the matrix X
%    x = xi';                        % get the transpose of xi for θTx
%    hthetai = theta' * x;           % Compute the hθ(x)=θTx for the ith training set
%    temp = hthetai - y([i],:);      % get the actual result for the ith training set

%    sumJ = sumJ + (temp*temp);      % the sum of squares
%    sumGrad = sumGrad + temp
%end

%fprintf('Size of X :  %f  \n', size(X));
%fprintf('Size of theta :  %f  \n', size(theta));

% vectorized form of above loop for calculating cost
htheta = X*theta;
hthetaMinusY = htheta - y;
sumJ = hthetaMinusY' * hthetaMinusY;
J = sumJ / (2 * m);

% calculating the regularized cost
theta1 = theta(2:size(theta,1), :);     %remove theta0
thetaJsquare = theta1'*theta1;
J = J + (lambda * thetaJsquare) / (2 * m);

% for calculating the gradient
sumGrad = X' * hthetaMinusY;
grad = sumGrad / m;

% calculating the regularized gradient
thetaVector = theta1 .* (lambda/m);     % calculate the regularised theta vector , excluding theta0
grad = grad + [0;thetaVector];          % vector addition, add the theta0 = 0, since we dont regularise the theta0

% =========================================================================

grad = grad(:);

end
