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
m = size(X, 2);

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

% mean/std automatically calculates the values for each feature

% set mu to the mean of X
mu = mean(X);
% set sigma to the standard deviation of X
sigma = std(X);


% iterate over every feature
for i = 1:m

% set cx to current feature based on iterator
cx = X(:,i);

% subtrat mean of cx from each value of cx
cx = cx - mu(i);

% divide each value on cx by standard deviation
cx = cx / sigma(i);

X_norm(:,i) = cx;

end

% ============================================================

end
