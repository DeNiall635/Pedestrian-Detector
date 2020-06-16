%% Fixed partition evaluation of EDGES-HOG using KNN classification with k values of 1, 3, 5, 10

clear all;
close all;

load('trainingDataPreprocessed.mat');
load('trainEdgeFeatures');

imDim = [160 96];

kValues = [1,3,5,10];

edgeHogs = getHogs(trEdges, imDim);

% Create dataset and shuffle
trData = horzcat(trLabels, edgeHogs);
trData = trData(randperm(size(trData,1)),:);

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

% arrays for raw metrics
Totals = [];
TPs = [];
TNs = [];
FPs = [];
FNs = [];

% vectors to store evaluation

kVals = [];
accuracies = [];
errorRates = [];
sensitivities = [];
precisions = [];
specificities = [];
falseAlarmRates = [];
fMeasures = [];
aucs = [];
cTimes = [];
for k = 1:size(kValues,2)
    kValue = kValues(k);
    
    tic
    
    % create model using reduced edge images with current c value
    knnModel = fitcknn(Xpca, trainLabels, 'NumNeighbors', kValue);
    
    modelPredictions = zeros(1,tsSize);
    for i = 1:tsSize
        % no need to reshape or calculate hogs - they are already calculated
        tsImEdge = (testImages(i,:) - meanX) * eigenVectors;
        
        [label, score] = predict(knnModel, tsImEdge);
        modelPredictions(1,i) = label;
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
    
    kVals = [kVals; kValue];
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

% plot legend for c values
hold on
lgd = legend({'1','3','5','10'},'Location','southeast');
title(lgd, 'K');
hold off

saveas(gcf, strcat('reportData/rocCurves/EDGE_KNN - Fixed Partition ROC Curve.png'));

figure(2)

% set table data
tblHeadings = {"NN", "Accuracy", "Error Rate", "Sensitivity", "Precision", "Specificity", "False Alarm Rate", "F-Measures", "AUC", "Time"};
tblData = horzcat(kVals, accuracies, errorRates, sensitivities, precisions, specificities, falseAlarmRates, fMeasures, aucs, cTimes);
uitable("Data", tblData, "ColumnName", tblHeadings);
disp(tblData);

% save the table into a csv .txt
tbl = table(kVals, accuracies, errorRates, sensitivities, precisions, specificities, falseAlarmRates, fMeasures, aucs, cTimes);
tbl.Properties.VariableNames = {'NN', 'Accuracy', 'Error Rate', 'Sensitivity', 'Precision', 'Specificity', 'False Alarm Rate', 'F-Measures', 'AUC', 'Time'};
writetable(tbl,'reportData/evaluationResults/edge_knn_fixed_partition','Delimiter',',', 'WriteRowNames', true);

% save raw metrics into a csv .txt
tbl = table(kVals, Totals, TPs, TNs, FPs, FNs);
tbl.Properties.VariableNames = {'NN', 'No. Observations','TP','TN','FP','FN'};
writetable(tbl, 'reportData/evaluationResults/rawResults/edge_knn_fixed_partition_raw', 'Delimiter', ',', 'WriteRowNames', true);