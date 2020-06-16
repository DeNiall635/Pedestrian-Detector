%% Perform K-fold cross validation using the HOG-BOOST-DT combination with k values of 3, 5, 10
% Different Learning Cycles of 50, 100 & 150 will also be applied
% We will observe the accuracy of the different values of k to strike a
% balance between performance and accuracy.

clear all;
close all;

load('trainingDataPreprocessed.mat');
load('trainHogFeaturesPreprocessed.mat');
imDim = [160 96];

% initialise Learning Cycle values
lValues = [50,100,150];

% initialise fold values
foldValues = [3,5,10];

% shuffle dataset
trainingDataset = horzcat(trLabels, trHogs);
trainingDataset = trainingDataset(randperm(size(trainingDataset,1)),:);

datasetSize = size(trainingDataset,1);

% arrays for raw metrics
Totals = [];
TPs = [];
TNs = [];
FPs = [];
FNs = [];

% arrays for calculated metrics
lcValues = [];
nFold = [];
accuracies = [];
errorRates = [];
sensitivities = [];
precisions = [];
specificities = [];
falseAlarmRates = [];
fMeasures = [];
aucs = [];
cTimes = [];

% for each Learning Cycle Value
for l = 1:length(lValues)
    
    nlearn = lValues(l);
    
    % run n fold cross validation on the data
    sprintf("%i Learning Cycle", nlearn)
    
    for f = 1:length(foldValues)
        
        % set number of folds
        nFolds = foldValues(f);
        
        % split the dataset into training and test datasets
        testSize = round(datasetSize / nFolds);
        trainSize = datasetSize - testSize;
        
        sprintf("%i fold cv", nFolds)
        
        clf
        % set the name of the ROC curve for this k value / cross validation
        rocName = sprintf("HOG_BOOST_DT - %i Learning Cycle - %i fold cv - ROC Curve", nlearn, nFolds);
        
        for i = 1:testSize:datasetSize
            
            foldTrainData = [];
            foldTestData = [];
            
            % break if we reach the end of the dataset
            if (i == datasetSize)
                break
            end
            
            % indicate test rows this fold
            foldTestRows = i:i+testSize-1;
            
            % if we are at the beginning, then only add rows after test data
            if (i == 1)
                foldTrainRows = [max(foldTestRows) + 1:datasetSize];
                % otherwise, we take rows before and after
            else
                foldTrainRows = [1:i-1 max(foldTestRows) + 1:datasetSize];
            end
            
            % load datasets from selected rows
            foldTrainData = [foldTrainData; trainingDataset(foldTrainRows,:)];
            foldTestData = [foldTestData; trainingDataset(foldTestRows,:)];
            
            % load images and labels from dataset
            trainingImages = foldTrainData(:,[2:size(foldTrainData,2)]);
            trainingLabels = foldTrainData(:,1);
            
            testImages = foldTestData(:,[2:size(foldTestData,2)]);
            testLabels = foldTestData(:,1);
            
            % reshape for comparison with predictions later
            testLabels = reshape(testLabels, 1, size(testLabels,1));
            
            % apply dimensionality reduction using pca to the hog features
            [eigenVectors, eigenValues, meanX, Xpca] = PrincipalComponentAnalysis(trainingImages);
            
            tic
            
            % createModel
            BoostModel = fitensemble(Xpca, trainingLabels, 'AdaBoostM1', nlearn, 'Tree');
            
            modelPredictions = zeros(1,testSize);
            
            for j = 1:size(testImages,1)
                tsImHog = (testImages(j,:) - meanX) * eigenVectors;
                [label, score] = predict(BoostModel,tsImHog);
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
            
            % plot roc curve for this fold
            hold on
            auc = plotROC(testLabels, modelPredictions);
            hold off
            
            f1 = (2 * sensitivity * precision) / (sensitivity + precision);
            
            % append metrics to corresponding arrays (final results)
            
            lcValues = [lcValues; nlearn];
            nFold = [nFold; nFolds];
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
        
        % set legend headings
        lgdHd = strings([1,nFolds]);
        for l = 1:nFolds 
            lgdHd(l) = num2str(l);
        end
        
        % plot legend
        hold on
        lgd = legend(lgdHd, 'Location', 'southeast');
        title(lgd, 'Fold No.');
        hold off
        
        % save for future reference
        saveas(gcf,strcat('reportData/rocCurves/', rocName, '.png'));
        
    end
end

figure(2)

% display evaluation metrics
tblHeadings = {"Learning Cycle", "N Folds", "Accuracy", "Error Rate", "Sensitivty", "Precision", "Specificity", "False Alarm Rate", "F-Measures", "AUC", "Time"};
tblData = horzcat(lcValues, nFold, accuracies, errorRates, sensitivities, precisions, specificities, falseAlarmRates, fMeasures, aucs, cTimes);
uitable("Data", tblData, "ColumnName", tblHeadings);
disp(tblData);

% save evaluation metrics into a csv .txt
tbl = table(lcValues, nFold, accuracies, errorRates, sensitivities, precisions, specificities, falseAlarmRates, fMeasures, aucs, cTimes);
tbl.Properties.VariableNames = {'Learning Cycle', 'N Folds', 'Accuracy', 'Error Rate', 'Sensitivty', 'Precision', 'Specificity', 'False Alarm Rate', 'F-Measures', 'AUC', 'Time'};
writetable(tbl,'reportData/evaluationResults/hog_boost_dt_cross-validated','Delimiter',',', 'WriteRowNames', true);

% save raw metrics into a csv .txt
tbl = table(lcValues, nFold, Totals, TPs, TNs, FPs, FNs);
tbl.Properties.VariableNames = {'Learning Cycle', 'N Folds', 'No. Observations','TP','TN','FP','FN'};
writetable(tbl, 'reportData/evaluationResults/rawResults/hog_boost_dt_cross-validated_raw', 'Delimiter', ',', 'WriteRowNames', true);
