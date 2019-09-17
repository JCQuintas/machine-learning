function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Prepare a1 to have a bias
a1 = [ones(m,1) X];

% Calculate z2 as the product of a1 by theta1 transpose
z2 = a1 * Theta1';

% Prepare a2 to have bias
% and the sigmoid of z2
a2 = [ones(size(z2),1) sigmoid(z2)];

% Calculate z3 as the product of a2 by theta2'
z3 = a2 * Theta2';

% a3 is simply the sigmoid of z3
a3 = sigmoid(z3);

% Get the maximum value for each row
[predicted_max, index_max] = max(a3, [], 2);

% Return the index of the maximum value, one of 
% 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
% each indicates which value was predicted, with 10 being 0
p = index_max;

% =========================================================================


end
