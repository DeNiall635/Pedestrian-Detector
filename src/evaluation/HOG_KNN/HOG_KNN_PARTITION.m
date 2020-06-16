%% Fixed partition evaluation of KNN classification with k values of 1, 3, 5, 10

clear all;
close all;

load('trainingDataPreprocessed.mat');
load('trainHogFeaturesPreprocessed.mat');

imDim = [160 96];

% initialise k values
kValues = [1,3,5,10];

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

% apply pca
[eigenVectors, eigenValues, meanX, Xpca] = PrincipalComponentAnalysis(trainImages);

% evaluate model accuracy for the different values of k

% arrays for raw metrics
Totals = [];
TPs = [];
TNs = [];
FPs = [];
FNs = [];

% arrays for calculated metrics

nnValues = [];
accuracies = [];
errorRates = [];
sensitivities = [];
precisions = [];
specificities = [];
falseAlarmRates = [];
fMeasures = [];
aucs = [];
cTimes = [];

for i = 1:length(kValues)
    k = kValues(i)
    
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
    [fResults, rResults] = evaluateModel(modelPredictions, testLabels);
    
    accuracy = fResults.accuracy;
    specificity = fResults.specificity;
    sensitivity = fResults.sensitivity;
    precision = fResults.precision;
    
    % calculate error rate, false alarm rate, area-under-curve and
    % f-measure
    errorRate = 1-accuracy;
    falseAlarmRate = 1-specificity;
    
    hold on
    auc = plotROC(testLabels, modelPredictions);
    hold off
    
    f1 = (2 * sensitivity * precision) / (sensitivity + precision);
    
    % append metrics to corresponding arrays
    
    nnValues = [nnValues; k];
    accuracies = [accuracies; accuracy];
    errorRates = [errorRates; errorRate];
    sensitivities = [sensitivities; sensitivity];
    precisions = [precisions; precision];
    specificities = [specificities; specificity];
    falseAlarmRates = [falseAlarmRates; falseAlarmRate];
    fMeasures = [fMeasures; f1];
    aucs = [aucs; auc];
    cTimes = [cTimes; cTime];
    
    % append raw metrics to corresponding arrays
    Totals = [Totals; rResults.Total];
    TPs = [TPs; rResults.TP];
    TNs = [TNs; rResults.TN];
    FPs = [FPs; rResults.FP];
    FNs = [FNs; rResults.FN];
    
end

% plot the legend
hold on
lgd = legend({'1','3','5','10'},'Location', 'southeast');
title(lgd, 'K Value');
hold off

% save to file
saveas(gcf, 'reportData/rocCurves/HOG_KNN - Fixed Partition.png');

figure(2)

% set table data
tblHeadings = {"NN", "Accuracy", "Error Rate", "Sensitivty", "Precision", "Specificity", "False Alarm Rate", "F-Measures", "AUC", "Time"};
tblData = horzcat(nnValues,accuracies, errorRates, sensitivities, precisions, specificities, falseAlarmRates, fMeasures, aucs, cTimes);
uitable("Data", tblData, "ColumnName", tblHeadings);
disp(tblData);

% save the table into a csv .txt
tbl = table(nnValues,accuracies, errorRates, sensitivities, precisions, specificities, falseAlarmRates, fMeasures, aucs, cTimes);
tbl.Properties.VariableNames = {'NN', 'Accuracy', 'Error Rate', 'Sensitivty', 'Precision', 'Specificity', 'False Alarm Rate', 'F-Measures', 'AUC', 'Time'};
writetable(tbl,'reportData/evaluationResults/hog_knn_fixed_partition','Delimiter',',', 'WriteRowNames', true);

% save raw metrics into a csv .txt
tbl = table(nnValues, Totals, TPs, TNs, FPs, FNs);
tbl.Properties.VariableNames = {'NN', 'No. Observations','TP','TN','FP','FN'};
writetable(tbl, 'reportData/evaluationResults/rawResults/hog_knn_fixed_partition_raw', 'Delimiter', ',', 'WriteRowNames', true);
