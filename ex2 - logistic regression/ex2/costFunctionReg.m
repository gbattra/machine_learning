function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta); % number theta's

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

predictions = X * theta;
predictions = sigmoid(predictions);
cost = 0;
for i = 1:m
  J = J + (-y(i) * log(predictions(i)) - (1 - y(i)) * log(1 - predictions(i)));
  cost = cost + (predictions(i) - y(i)) * X(i,:)';
end

theta_square_sum = 0;
for i = 2:n
  theta_square_sum = theta_square_sum + theta(i)^2;
end

J = (1 / m) * J;
J = J + (lambda / (2 * m)) * theta_square_sum;

for i = 1:n
   grad(i) = (1 / m) * cost(i);
   if i > 1
     grad(i) = grad(i) + (lambda / m) * theta(i);
   end
end  

% =============================================================

end
