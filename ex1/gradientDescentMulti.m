function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
f = size(X, 2);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %


    % we copy theta so we don't update the real value by accident
    t_ = theta;

    % Define h as the product of X times theta
    h = X * theta;

    % iterate over the features of X
    for j = 1:f;

      % small x is just the current 'feature' (j) column
      x =  X(:,j);

      % for each feature, calculate the new theta(j)
      t_(j) = t_(j) - alpha * (1/m) * sum((h - y) .* x);

    end

    % asign new theta to be used on next iteration or returned
    theta = t_;


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end

