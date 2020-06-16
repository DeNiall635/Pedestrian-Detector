%% evaluates the model predictions against the test dataset and returns results as separate structs.
% fResults - final results including accuracy, sensitivity etc.
% rResults - raw results i.e. total test observations, true positives etc.
function [fResults, rResults] = evaluateModel(modelPredictions, testLabels)

    truePositive = 0;
    falsePositive = 0;
    trueNegative = 0;
    falseNegative = 0;
    
    for i=1:size(modelPredictions, 2)
        testResult = modelPredictions(1, i);
        actualResult = testLabels(1, i);
        if testResult == 1 && actualResult == 1
            truePositive = truePositive + 1;
        elseif testResult == -1 && actualResult == -1
            trueNegative = trueNegative + 1;
        elseif testResult == 1 && actualResult == -1
            falsePositive = falsePositive + 1;
        elseif testResult == -1 && actualResult == 1
            falseNegative = falseNegative + 1;
        end
    end
    
    rResults.Total = size(modelPredictions,2);
    rResults.TP = truePositive;
    rResults.FP = falsePositive;
    rResults.TN = trueNegative;
    rResults.FN = falseNegative;
    
    fResults.accuracy = (truePositive + trueNegative) / size(testLabels, 2);
    fResults.sensitivity = truePositive / (truePositive + falseNegative);
    fResults.precision = truePositive / (truePositive + falsePositive);
    fResults.specificity = trueNegative / (trueNegative + falsePositive);
end