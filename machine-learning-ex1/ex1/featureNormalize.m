function [X_norm, mu, sigma] = featureNormalize(X)

%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       


% no of features available in the feature vector
no_of_features = size(X, 2);    % no of columns of X, since every column is a feature

% no of training sets, i.e. no of rows of X
m = size(X, 1);

% row vector containing the mean of all the columns
meanVector = mean(X);

for columnNo = 1 : no_of_features, %iterate for each feature/column of X

    % calculate the mean of the current feature
    mean  = meanVector(1, columnNo);
    mu(1, columnNo) = mean;

    %compute the standard deviation of each feature
    sdev = std(X(:, columnNo));
    sigma(1, columnNo) = sdev;

    for rowNo = 1 : m,
        fvalue = X(rowNo, columnNo);
        X_norm(rowNo, columnNo) =  (fvalue - mean) / sdev;
    end

end


% ============================================================

end
