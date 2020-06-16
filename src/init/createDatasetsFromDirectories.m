%% Load the files in the training and test directories into reusable workspaces.

% Clear current workspace variables
clear all;
% Close open files, figures and subplots
close all;

% choose training directories (2 folders for positive and negative)
trPosDir = uigetdir;
trNegDir = uigetdir;

% Create database objects for positive and negative samples, setting the
% labels for each.
[imPos,posLabels] = createDatabaseFromDirectory(trPosDir,1);
[imNeg,negLabels] = createDatabaseFromDirectory(trNegDir,-1);

% combine pos and neg databases into single training database.
trIm = [imPos; imNeg];
trLabels = [posLabels;negLabels];

% choose test directory
tsDir = uigetdir;

% create database objects for test data, set labels to 0 since we have the
% .cdataset with proper labelling.
[tsImg,tsLabels] = createDatabaseFromDirectory(tsDir,0);

% save databases into .mat files for future use.
save('trainingData','trIm','trLabels');
save('testData','tsImg');