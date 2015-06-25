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

	File name:    Matlab_FeatureBoxplots.m
	Created:      October 30th, 2014
	Author:       Rob Lyon
 
	Contact:    rob@scienceguyrob.com or robert.lyon@postgrad.manchester.ac.uk
	Web:        <http://www.scienceguyrob.com> or <http://www.cs.manchester.ac.uk> 
            or <http://www.jb.man.ac.uk>
            
    Example showing how to build box plots showing feature class separation.
    
%}

% Clear matlab workspace.
clear

% First load the data from a CSV file.
% Here I'm loading the test and training data from a single file.
% data_X contains the data, and data_Y the correct class labels.
file = '~/Dropbox/Documents/SharedWorkspace/JACS_ML_Examples/data/magic.csv';
data = csvread(file');

samples  = size(data,1);
features = size(data,2);

% We need the class labels so we can separate positive & negative classes.
labels = data(:,features);
positive_examples = sum(labels);
negative_examples = samples - positive_examples;

% Stores data for each class separately.
positives = zeros(positive_examples,features);
negatives = zeros(negative_examples,features);

posCount = 0;
negCount = 0;

for row = 1:samples
    
    % Obtain a n example...
    rd = data(row,:);
    
    % If positive store in the positive data matrix.
    if labels(row,1)==1
        posCount=posCount+1;
        positives(posCount,:) = rd ;
    % If negative store in the negative data matrix.
    else
        negCount=negCount+1;
        negatives(negCount,:) = rd ;
    end;

end;

% This next section creates the box plots of the data.
% Change these settings here to produce box plots for different features.
featureToStudy = 6;
positive_data = positives(:,featureToStudy);
negative_data = negatives(:,featureToStudy);
description = strcat('Feature ',num2str(featureToStudy),' class separability');

group = [repmat({'Negative'},negative_examples,1) ; repmat({'Positive'},positive_examples,1)];
boxplot([negative_data;positive_data], group)
%set(gca,'YScale','log')

% Construct plot title.      
str1 = 'Box plot of ' ;
title(strcat(str1,description));

% Label the axes.
xlabel('Class')
ylabel('Value')

% OBTAIN STATS INFO FOR PLOT.

% Mean
posMean = mean(positive_data);
negMean = mean(negative_data);

% Min
posMin = min(positive_data);
negMin = min(negative_data);

% Max
posMax = max(positive_data);
negMax = max(negative_data);

% Median
posMedian = median(positive_data);
negMedian = median(negative_data);

% STDEV
posSTDEV = std(positive_data);
negSTDEV = std(negative_data);

% Q1
posQ1 = quantile(positive_data,0.25);
negQ1 = quantile(negative_data,0.25);

% Q3
posQ3 = quantile(positive_data,0.75);
negQ3 = quantile(negative_data,0.75);

% IQR
posIQR = posQ3-posQ1;
negIQR = negQ3-negQ1;

% Range
posRange = posMax-posMin;
negRange = negMax-negMin;

% Cacluate the Y co-ordinate to draw labels to:
%y = max([negMax,posMax,predMax])/2
y = max([negMax,posMax]);
padd = (max([negMax,posMax])/9)/2;

% NEGATIVE PLOT LABELS
negX =0.75;
text(negX,y,'Negative Class','HorizontalAlignment','center');
text(negX,y-padd,strcat('Median: ',num2str(negMedian)),'HorizontalAlignment','center');
text(negX,y-2*padd,strcat('Min: ',num2str(negMin)),'HorizontalAlignment','center');
text(negX,y-3*padd,strcat('Max: ',num2str(negMax)),'HorizontalAlignment','center');
text(negX,y-4*padd,strcat('Q1: ',num2str(negQ1)),'HorizontalAlignment','center');
text(negX,y-5*padd,strcat('Q3: ',num2str(negQ3)),'HorizontalAlignment','center');
text(negX,y-6*padd,strcat('IQR: ',num2str(negIQR)),'HorizontalAlignment','center');
text(negX,y-7*padd,strcat('Range: ',num2str(negRange)),'HorizontalAlignment','center');
text(negX,y-8*padd,strcat('Stdev: ',num2str(negSTDEV)),'HorizontalAlignment','center');

% POSATIVE PLOT LABELS
posX =1.75;
text(posX,y,'Positive Class','HorizontalAlignment','center');
text(posX,y-padd,strcat('Median: ',num2str(posMedian)),'HorizontalAlignment','center');
text(posX,y-2*padd,strcat('Min: ',num2str(posMin)),'HorizontalAlignment','center');
text(posX,y-3*padd,strcat('Max: ',num2str(posMax)),'HorizontalAlignment','center');
text(posX,y-4*padd,strcat('Q1: ',num2str(posQ1)),'HorizontalAlignment','center');
text(posX,y-5*padd,strcat('Q3: ',num2str(posQ3)),'HorizontalAlignment','center');
text(posX,y-6*padd,strcat('IQR: ',num2str(posIQR)),'HorizontalAlignment','center');
text(posX,y-7*padd,strcat('Range: ',num2str(posRange)),'HorizontalAlignment','center');
text(posX,y-8*padd,strcat('Stdev: ',num2str(posSTDEV)),'HorizontalAlignment','center');
