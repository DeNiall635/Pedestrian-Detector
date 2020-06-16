%% Fixed partition evaluation of SVM classification with c values of 0.1, 0.25, 0.5, 1

clear all;
close all;

load('trainingDataPreprocessed.mat');
load('trainHogFeaturesPreprocessed.mat');
imDim = [160 96];

cValues = [0.1,0.25,0.5,1];

% Create dataset and shuffle
trData = horzcat(trLabels, trHogs);
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

cVals = [];
accuracies = [];
errorRates = [];
sensitivities = [];
precisions = [];
specificities = [];
falseAlarmRates = [];
fMeasures = [];
aucs = [];
cTimes = [];

for c = 1:length(cValues)
    cValue = cValues(c);
    
    tic
    
    % create model using reduced hogs with current c value
    svmModel = fitcsvm(Xpca, trainLabels, 'KernelFunction', 'rbf', 'BoxConstraint', cValue);
    
    modelPredictions = zeros(1,tsSize);
    for i = 1:tsSize
        % no need to reshape or calculate hogs - they are already calculated
        tsImHog = (testImages(i,:) - meanX) * eigenVectors;
        
        [label, score] = predict(svmModel, tsImHog);
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
    
    cVals = [cVals; cValue];
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
lgd = legend({'0.1','0.25','0.5','1'},'Location','southeast');
title(lgd, 'C Value');
hold off

saveas(gcf, strcat('reportData/rocCurves/HOG_SVM - Fixed Partition ROC Curve.png'));

figure(2)

% set table data
tblHeadings = {"C", "Accuracy", "Error Rate", "Sensitivity", "Precision", "Specificity", "False Alarm Rate", "F-Measures", "AUC", "Time"};
tblData = horzcat(cVals, accuracies, errorRates, sensitivities, precisions, specificities, falseAlarmRates, fMeasures, aucs, cTimes);
uitable("Data", tblData, "ColumnName", tblHeadings);
disp(tblData);

% save the table into a csv .txt
tbl = table(cVals, accuracies, errorRates, sensitivities, precisions, specificities, falseAlarmRates, fMeasures, aucs, cTimes);
tbl.Properties.VariableNames = {'C', 'Accuracy', 'Error Rate', 'Sensitivity', 'Precision', 'Specificity', 'False Alarm Rate', 'F-Measures', 'AUC', 'Time'};
writetable(tbl,'reportData/evaluationResults/hog_svm_fixed_partition','Delimiter',',', 'WriteRowNames', true);

% save raw metrics into a csv .txt
tbl = table(cVals, Totals, TPs, TNs, FPs, FNs);
tbl.Properties.VariableNames = {'C', 'No. Observations', 'TP', 'TN', 'FP', 'FN'};
writetable(tbl, 'reportData/evaluationResults/rawResults/hog_svm_fixed_partition_raw', 'Delimiter', ',', 'WriteRowNames', true);

