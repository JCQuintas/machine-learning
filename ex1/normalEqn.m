function [theta] = normalEqn(X, y)
%NORMALEQN Computes the closed-form solution to linear regression 
%   NORMALEQN(X,y) computes the closed-form solution to linear 
%   regression using the normal equations.

theta = zeros(size(X, 2), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the code to compute the closed form solution
%               to linear regression and put the result in theta.
%

% ---------------------- Sample Solution ----------------------

% This is basically of lecture 4's solution.
% Pseudo Inverse of the result X transpose times X
% Times X transpose times y

theta = pinv(X' * X) * X' * y;

% -------------------------------------------------------------


% ============================================================

end
