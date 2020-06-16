%% Perform K-fold cross validation using the HOG-SVM combination with k values of 3, 5, 10
% We will observe the accuracy of the different values of k to strike a
% balance between performance and accuracy.

% initialise values of k
foldValues = [3,5,10];

% initialise values of c
cVals = [0.1,0.25,0.5,1];

imDim = [160 96];

% load the dataset
load("trainingDataPreprocessed");

% since we have pre-computed the hogs, we can just load them and use them
% in place of the image data - reduce computation time
load('trainHogFeaturesPreprocessed');

% shuffle dataset - pass in the pre-computed hogs
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
cVals = [];
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

% for every c value, perform k fold cross validation on an svm model
for c = 1:length(cValues)
    % get c value as parameter for model
    cValue = cValues(c);
    
    sprintf('C - %i', cValue)
    
    for h = 1:size(foldValues,2)
        % set value of k
        nFolds = foldValues(h);
        
        % used to log progress to console
        sprintf('%i fold cv', nFolds)
        
        
        
        % split the dataset into training and test datasets
        testSize = round(datasetSize / nFolds);
        trainSize = datasetSize - testSize;

        clf
        
        % the name of the roc curve .png that will be save
        graphName = sprintf('HOG_SVM - c = %i - %i Fold CV - ROC Curve', cValue, nFolds);
        
        % create datasets for this fold
        for i = 1:testSize:datasetSize
            
            foldTrainData = [];
            foldTestData = [];
                        
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
            
            % start timer
            tic
            
            % apply dimensionality reduction using pca to the hog features
            [eigenVectors, eigenValues, meanX, Xpca] = PrincipalComponentAnalysis(trainingImages);
            
            % create model
            svmModel = fitcsvm(Xpca, trainingLabels, 'KernelFunction', 'rbf', 'BoxConstraint', cValue);
            
            modelPredictions = zeros(1,testSize);
            
            % predict labels of test data using model
            for j = 1:size(testImages,1)
                % get the test image hog data
                
                % calculate hog for test image -> remember to use results from
                % pca such that it can used in predictions.
                % (feature - meanX) * eigenVectors
                tsImHog = (testImages(j,:) - meanX) * eigenVectors;
                [label, score] = predict(svmModel,tsImHog);
                
                % store prediction
                modelPredictions(1,j) = label;
            end
 
            cTime = toc;
            
            % get evaluation metrics
            [fResults, rResults] = evaluateModel(modelPredictions, testLabels);
            
            accuracy = fResults.accuracy;
            specificity = fResults.specificity;
            sensitivity = fResults.sensitivity;
            precision = fResults.precision;       

            
            errorRate = 1 - accuracy;
            falseAlarmRate = 1 - specificity;
            
            f1 = (2 * sensitivity * precision) / (sensitivity + precision);
            
            % plot roc curve for this fold
            hold on
            auc = plotROC(testLabels, modelPredictions);
            hold off
            
            % add new entries to data
            cVals = [cVals; cValue];
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
        
        saveas(gcf,strcat('reportData/rocCurves/', graphName, '.png'));
    end
    
end



% set table data
tblHeadings = {"C", "No. Folds", "Accuracy", "Error Rate", "Sensitivty", "Precision", "Specificity", "False Alarm Rate", "F-Measure", "AUC", "Time"};
tblData = horzcat(cVals, nFold, accuracies, errorRates, sensitivities, precisions, specificities, falseAlarmRates, fMeasures, aucs, cTimes);
uitable("Data", tblData, "ColumnName", tblHeadings);
disp(tblData);

tbl = table(cVals, nFold, accuracies, errorRates, sensitivities, precisions, specificities, falseAlarmRates, fMeasures, aucs, cTimes);
tbl.Properties.VariableNames = {'C', 'No. Folds', 'Accuracy', 'Error Rate', 'Sensitivty', 'Precision', 'Specificity', 'False Alarm Rate', 'F-Measure', 'AUC', 'Time'};
writetable(tbl,'reportData/evaluationResults/hog_svm_cross-validated','Delimiter',',', 'WriteRowNames', true);

% save raw metrics into a csv .txt
tbl = table(cVals, nFold, Totals, TPs, TNs, FPs, FNs);
tbl.Properties.VariableNames = {'C', 'N Folds', 'No. Observations','TP','TN','FP','FN'};
writetable(tbl, 'reportData/evaluationResults/rawResults/hog_svm_cross-validated_raw', 'Delimiter', ',', 'WriteRowNames', true);

