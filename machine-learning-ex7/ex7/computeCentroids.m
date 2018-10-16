function centroids = computeCentroids(X, idx, K)

%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%


for centroid = 1 : K

    % gives a m dimensional vector with value 1 and 0 s. The indexes where the value is 1 , are the Points which are assigned to that centroid.
    indexes = (idx == centroid);

    % below should give sum of the points for which indexes==1
    summation = indexes' * X;

    % total no of points which are assigned to the centroid, basically just sum all the 1s in indexes
    count = sum(indexes);

    centroids(centroid, :) = summation/count;

end

% =============================================================


end

