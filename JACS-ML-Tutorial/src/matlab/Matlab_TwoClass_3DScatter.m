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

	File name:    Matlab_TwoClass_3DScatter.m
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
posFile = '/Users/rob/Dropbox/Documents/SharedWorkspace/JACS_ML_Examples/data/magic_positive.csv';
negFile = '/Users/rob/Dropbox/Documents/SharedWorkspace/JACS_ML_Examples/data/magic_negative.csv';

positives = csvread(posFile);
negatives = csvread(negFile);

% Choose which data columns to plot.
xIndex = 1;
yIndex = 2;
zIndex = 3;

pos_X = positives(:, xIndex);
pos_Y = positives(:, yIndex);
pos_Z = positives(:, zIndex);

neg_X = negatives(:, xIndex);
neg_Y = negatives(:, yIndex);
neg_Z = negatives(:, zIndex);


% Construct plot title               
t = strcat('Feature ',num2str(xIndex),' (x-axis) vs Feature ', num2str(yIndex),' (y-axis) vs Feature ', num2str(zIndex),' (z-axis).');

scatter3(pos_X,pos_Y,pos_Z);

%set(gca,'YScale','log');
%set(gca,'XScale','log');
%set(gca,'ZScale','log');

hold on

scatter3(neg_X,neg_Y,neg_Z);
%set(gca,'YScale','log');
%set(gca,'XScale','log');
%set(gca,'ZScale','log');

hold off

legend('+','-');
title(t);
xlabel(strcat('Feature ',num2str(xIndex),' .'));
ylabel(strcat('Feature ',num2str(yIndex),' .'));
zlabel(strcat('Feature ',num2str(zIndex),' .'));