function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

% Calculate h using the sigmoid function
h = sigmoid(X * theta);

% We don't want to regularize theta 0
% So we set it to 0, since it's used in an addition
regularized_theta = [0; theta(2: end)];

% Calculate the regularization
regularization = lambda / (2 * m) * sum(regularized_theta .^ 2);

% Define the cost J using appropriate formula
J = 1 / m * sum(-y .* log(h) - (1 - y) .* log(1 - h)) + regularization;

% Define gradient using new 'h of theta' formula
% And also add a regularization part
grad = 1 / m * sum((h - y) .* X) + lambda / m * regularized_theta;

% =============================================================

end
