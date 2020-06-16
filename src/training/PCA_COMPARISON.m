%% Evaluate the difference in computation time and accuracy when applying PCA to a HOG_KNN classifier

clear all;
close all;

load('trainingDataPreprocessed.mat');
load('trainHogFeaturesPreprocessed.mat');

k = 5;
imDim = [160 96];

% Create dataset and shuffle
trData = horzcat(trLabels, trHogs);
trData = trData(randperm(size(trData,1)),:);

% split the data: 75 percent training, 25 percent testing

trSize = round(size(trData,1)* 0.75);
tsSize = size(trData,1) - trSize;

% create training and test datasets
trainDataset = trData(1:trSize,:);
trainImages = trainDataset(:,[2:size(trainDataset,2)]);
trainLabels = trainDataset(:,1);

testDataset = trData(trSize+1:end,:);
testImages = testDataset(:,[2:size(testDataset,2)]);
testLabels = testDataset(:,1);
testLabels = reshape(testLabels,1,tsSize);

% table data

usePCA = [];
accuracies = [];
cTimes = [];

% WITHOUT PCA

tic

% fit the model using reduced hogs
knnModel = fitcknn(trainImages, trainLabels, 'NumNeighbors', k);

modelPredictions = zeros(1,tsSize);
for j = 1:tsSize
    % retrieve the test image and reduce dimensions
    tsImHog = testImages(j,:);
    
    % let the model predict its label
    [label, score] = predict(knnModel, tsImHog);
    modelPredictions(1,j) = label;
end

cTime = toc;

[accuracy, sensitivity, precision, specificity] = evaluateModel(modelPredictions, testLabels);

usePCA = [usePCA; "NO"];
accuracies = [accuracies; accuracy];
cTimes = [cTimes; cTime];

% WITH PCA

% apply PCA to HOG feature data

[eigenVectors, eigenValues, meanX, Xpca] = PrincipalComponentAnalysis(trainImages);

tic

% fit the model using reduced hogs
knnModel = fitcknn(Xpca, trainLabels, 'NumNeighbors', k);

modelPredictions = zeros(1,tsSize);
for j = 1:tsSize
    % retrieve the test image and reduce dimensions
    tsImHog = (testImages(j,:) - meanX) * eigenVectors;
    
    % let the model predict its label
    [label, score] = predict(knnModel, tsImHog);
    modelPredictions(1,j) = label;
end

cTime = toc;

% get evaluation metrics
[accuracy, sensitivity, precision, specificity] = evaluateModel(modelPredictions, testLabels);

usePCA = [usePCA; "YES"];
accuracies = [accuracies; accuracy];
cTimes = [cTimes; cTime];

% store results in table
tblHeadings = {'Use PCA?','Accuracy','Computation Time'};
tblData = horzcat(usePCA, accuracies, cTimes);
% uitable('Data', tblData, 'ColumnName', tblHeadings);
% disp(tblData);

% save the table into a csv .txt
tbl = table(usePCA, accuracies, cTimes);
tbl.Properties.VariableNames = tblHeadings;
writetable(tbl,'reportData/evaluationResults/pca_comparison','Delimiter',',', 'WriteRowNames', true);
