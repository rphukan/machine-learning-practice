function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)

%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

%%%%%%% calculating J with a loop
%for i = 1 : num_movies
%    for j = 1 : num_users
%        if R(i, j)==1
%            error = ( Theta(j, :) * X(i, :)' ) - Y(i, j);
%            J = J + error*error;
%        end
%    end
%end

%J = J/2;

%%%%%%%%%%%%%%%%%%%  vectorised calculation of cost J %%%%%%%%%%%%%%%%%%%
# predictions of movie ratings for each users θT * x
# size is (num_users  x num_features) * (num_movies  x num_features) = (num_users * num_movies)
predictions = Theta * X';

# the cost or error for each of these ratings, size(num_movies x num_users)
cost = predictions' - Y;

# take the cost for only those movies which user have rated
ratedcost = cost .* R;

# take square of the cost and sum it up for total cost
J = sum( sum(ratedcost .* ratedcost) ) / 2;

# calculating the regularization part
regularization = ((sum( sum(Theta .* Theta) ) + sum( sum(X .* X) )) * lambda )/ 2;

J = J + regularization;


%%%%%%%%%%%%%%%%%%% calculating the gradient %%%%%%%%%%%%%%%%%%%
% loop over all the movies
for i = 1 : num_movies
    % select the indexes for which this movie was rated by any user. This gives you a list of all the users that have rated movie i.
    % This is done because, when you consider the features for the i-th movie, you only need to be concern about the users who had
    % given ratings to the movie, and this allows you to remove all the other users from Theta and Y.
    idx = find(R(i, :) == 1);         % R - num_movies x num_users matrix
    Theta_temp = Theta(idx, :);       % Theta is num_users  x num_features, this gives Theta for only the set of users which have rated the i-th movie.
    Y_temp = Y(i, idx);               % Y is num_movies x num_users matrix, this gives Y only for the set of users which have rated the i-th movie.

    X_grad(i, :) = ( X(i, :) * Theta_temp' - Y_temp ) * Theta_temp + lambda .* X(i, :);
end

% loop over all the users
for j = 1 : num_users
    % list of all the movies that have been rated by the user j
    idx = find(R(:, j) == 1);          % R - num_movies x num_users matrix
    X_temp = X(idx, :);                % X - num_movies  x num_features, this gives X only for the set of movies which have been rated by the j th user
    Y_temp = Y(idx, j);                % Y is num_movies x num_users matrix, this gives Y only for the set of movies which have been rated by the j th user

    Theta_grad(j, :) = ( X_temp * Theta(j, :)' - Y_temp )' * X_temp + lambda .* Theta(j, :);
end

% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
