function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

% Calculate h using the sigmoid function
h = sigmoid(X * theta);

% Define the cost J using appropriate formula
J = 1 / m * sum(-y .* log(h) - (1 - y) .* log(1 - h));

% Define gradient using new 'h of theta' formula
grad = 1 / m * sum((h - y) .* X);

% =============================================================

end

% Cost at test theta: 14.400000
% Cost at test theta: 4.162757
% Cost at test theta: 4.369710
% Expected cost (approx): 0.218
% Gradient at test theta: 
%  -0.600000 
%  20.806955 
%  21.845107 
% Expected gradients (approx):
%  0.043
%  2.566
%  2.647