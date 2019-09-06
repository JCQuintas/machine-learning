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

% h(x) equals the result of the theta * X matrix multiplication
h = X * theta;

% intermediate variable to hold the value of the matrix subtration (h - y)
sub = h - y;

% intermediate variable to hold the value of applying a square root to 
% every value in the matrix (element-wise sqr)
sqr = sub .^ 2;

% cost(J) equals 1/2m times the sum of sqr
J = 1 / (2 * m) * sum(sqr);

% =========================================================================

end
