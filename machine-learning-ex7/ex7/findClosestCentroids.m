function idx = findClosestCentroids(X, centroids)

%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%
m = size(X,1);

for i = 1 : m

    xi = X(i, :);
    ci = -1;
    %fprintf(' \n xi is %d', xi);

    % compare with each centroid to get the nearest one
    mindist = -1;                   % the distance to the closest centroid
    for j = 1 : K
        muj = centroids(j, :);
        %fprintf(' \n muj is %d', muj);

        diff = xi - muj;            % the difference between the ith point xi and the jth centroid muj
        sqrdiff = diff .* diff;
        vector = ones(size(sqrdiff, 2), 1);
        distance = sqrdiff * vector;
        %fprintf(' \n distance is %d', distance);
        if mindist == -1
            mindist = distance;
            ci = j;
            %fprintf(' \n mindist  is %d', mindist);
        elseif distance < mindist
            mindist = distance;
            ci = j;
            %fprintf(' \n mindist  updated to  %d', mindist);
        end
    end

    idx(i, 1) = ci;
    %fprintf(' \n idx  is %d', idx);

end


% =============================================================

end

