function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
 Carr = [0.01,0.03,0.1,0.3,1,3,10,30]
 sigarr = [0.01,0.03,0.1,0.3,1,3,10,30]
 
row = 1
 finalarr = zeros(length( Carr)* length(sigarr),3)
 for i= 1:length(Carr)
   for j = 1:length(sigarr)
     currentC = Carr(i)
     currentsig = sigarr(j)
     model = svmTrain(X,y,currentC,@(x1,x2)gaussianKernel(x1,x2,currentsig))
     predictions = svmPredict(model, Xval);
     errorval =  mean(double(predictions ~= yval));
     finalarr(row,:) = [currentC currentsig errorval]
     row = row +1
  end
end
[val,ind]= min(finalarr(:,3))
C = finalarr(ind,1)
sigma = finalarr(ind,2)




% =========================================================================

end
