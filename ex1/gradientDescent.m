function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    % get all values from the second column of X
    x = X(:,2);

    % h(x) equals multiplying every entry on the x matrix by theta1 and then 
    % adding theta0 to each resulting value
    h = theta(1) + (theta(2)* x);

    % calculated value of theta0
    t0 = theta(1) - alpha * (1/m) * sum(h - y);

    % calculated value of theta1, almost the same as theta0 except we 
    % multiply each value by x(i) at the end
    t1 = theta(2) - alpha * (1/m) * sum((h - y) .* x);

    % update both values of theta at once
    theta = [t0; t1];

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
