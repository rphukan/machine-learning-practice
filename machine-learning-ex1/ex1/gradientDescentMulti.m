function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)

%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    tempTheta = zeros(size(theta, 1), 1);

    features = size(X, 2); % no of columns of X

    % calculate the theta for one by one feature

    for count = 1 : features,

        sum = 0;

        for i = 1 : m,

            %compute hθ(xi) first
            xi = X([i], :);
            x = xi';

            hthetai = theta' * x;

            % compute hθ(xi) - yi
            sum = sum + ((hthetai - y([i],:)) * x(count,1));

        end

        tempValue = theta(count,1) - (alpha*sum)/m;

        %fprintf(' tempValue %f \n', tempValue);

        tempTheta(count, 1) = tempValue;

        %fprintf(' tempTheta %f \n', tempTheta);

    end

    % ============================================================
    % Save the cost J in every iteration
    J_history(iter) = computeCost(X, y, theta);

    theta = tempTheta;
    %fprintf(' theta %f \n', theta);


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
