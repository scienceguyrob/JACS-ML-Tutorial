%{

	This file is part of the JACS ML Tutorial.

	JACS ML Tutorial is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	JACS ML Tutorial is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with JACS ML Tutorial.  If not, see <http://www.gnu.org/licenses/>.

	File name:    Matlab_DecisionTree_Example.m
	Created:      October 30th, 2014
	Author:       Rob Lyon
 
	Contact:    rob@scienceguyrob.com or robert.lyon@postgrad.manchester.ac.uk
	Web:        <http://www.scienceguyrob.com> or <http://www.cs.manchester.ac.uk> 
            or <http://www.jb.man.ac.uk>
            
    Example showing how to build a Decision tree classifier in Matlab.

%}

% Clear matlab workspace.
clear

% First load the data from a CSV file.
% Here I'm loading the test and training data from a single file.
% data_X contains the data, and data_Y the correct class labels.
file = '~/Dropbox/Documents/SharedWorkspace/JACS_ML_Examples/data/magic.csv';
orderedData = csvread(file');

% Randomly shuffle data.
data = orderedData(randperm(size(orderedData,1)),:);

samples  = size(data,1);
features = size(data,2);

% Create training and test data dats, by splitting the data.
k = 2; % The number of folds is 2 since 1 for training, and 1 for test.

% This generates indicies which identify which examples in the data set
% should be used for testing and which for training.
c = cvpartition(samples,'kfold',k);
trainingIndices = training(c,1);

train = [];
test  = [];

% This part is a fudge to create training and test sets.
% Matlab experts may know of an easier way to do this.
% PLEASE don't use this if you have large data sets, this 
% will eat your memory. Consider generating training/test data
% sets yourself outside of Matlab, or learn the Matlab API's
% if you have large (say over 100MB) of data.
for row = 1:samples
    
    % Choose the next example from the data set. Here the
    % training indices correspond to the test indices.
    choice = trainingIndices(row,1);
    example = data(row,:);
    
    % A choice equal to 1 means the example should be put in the
    % training set.
    if choice == 1
        train = [train;example];
    % A choice equal to 0 means the example should be put in the
    % test set.
    else
        test = [test;example];
    end;
    
end;

% Now we have training / test data.
train_X = train(:,1:features-1);
train_Y = train(:,features);

test_X  = test(:,1:features-1);
test_Y  = test(:,features);

% Build the KNN model
mdl = fitctree(train_X,train_Y);

% Make predictions on test data
predictions = predict(mdl,test_X);

% Evaluate the predictions.
TN = 0;
FN = 0;
FP = 0;
TP = 0;

for row = 1:size(predictions,1)
    
    prediction = trainingIndices(row,1);
    true_label = test_Y(row,1);
    
    if prediction ==  0             % Negative prediction.
        
        if prediction == true_label % Predicted negative, actual was negative.
            TN = TN + 1;
        else                        % Predicted negative, actual was positive.
            FN = FN + 1;
        end

    elseif prediction == 1          % Positive prediction.
        
        if prediction == true_label % Predicted positive, actual was positive.
            TP = TP + 1;
        else                        % Predicted positive, actual was negative.
            FP = FP + 1;
        end
    end
        
end;

% Calculate other metrics.
accuracy = (TP + TN) / (TP + FP + FN + TN);

precision = (TP) / (TP + FP);

recall = (TP) / (TP + FN);

specificity = (TN) / (FP+TN);

negativePredictiveValue = (TN) / (FN + TN);

matthewsCorrelation = ((TP * TN) - (FP * FN)) / sqrt((TP+FP) * (TP+FN) * (TN+FP) * (TN+FN));

fScore = 2 * ((precision * recall) / (precision + recall));
        
% Kappa = (totalAccuracy - randomAccuracy) / (1 - randomAccuracy)
%
% where,
%
% totalAccuracy = (TP + TN) / (TP + TN + FP + FN)
%
% and
%
% randomAccuracy = (TN + FP) * (TN + FN) + (FN + TP) * (FP + TP) / (Total*Total).
total     = TP + TN + FP + FN;
totalAcc  = (TP + TN) / (TP + TN + FP + FN);
randomAcc =  (((TN + FP) * (TN + FN)) + ((FN + TP) * (FP + TP))) / (total*total);
kappa = (totalAcc - randomAcc) / (1 - randomAcc);

gmean = sqrt( ( TP /( TP + FN ) ) * ( TN / ( TN + FP ) ) );

fprintf('\n****************************************\n')
fprintf('\nK-NN Performance.\n')
fprintf('TN : \t\t%i\n', TN)
fprintf('FN : \t\t%i\n', FN)
fprintf('FP : \t\t%i\n', FP)
fprintf('TP : \t\t%i\n', TP)
fprintf('Accuracy:\t%f\n', accuracy)
fprintf('Precision:\t%f\n', precision)
fprintf('Recall:\t\t%f\n', recall)
fprintf('Specificity:\t%f\n', specificity)
fprintf('NPV:\t\t%f\n', negativePredictiveValue)
fprintf('MCC:\t\t%f\n', matthewsCorrelation)
fprintf('F-Score:\t%f\n', fScore)
fprintf('Kappa:\t\t%f\n', kappa)
fprintf('G-Mean:\t\t%f\n', gmean)

% Maybe Visualise the data? Uncomment as you like.
%gscatter(train_X(:,1),train_X(:,2),train_Y) % View two variables in training data.
%gscatter(test_X(:,1),test_X(:,2),test_Y)% View two variables in test data.

% Visualise the training data in 3D.
scatter3(train_X(:,2),train_X(:,6),train_X(:,9),10,train_Y)