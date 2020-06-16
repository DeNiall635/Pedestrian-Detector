%% Evaluate the model and display the results
function displayEvaluationMetrics(modelPredictions, testLabels, figureNo)
    if (nargin < 3)
        figureNo = 1;
    end
    
    % evaluate model
    [accuracy, sensitivity, precision, specificity] = evaluateModel(modelPredictions, testLabels);
    
    figure(figureNo)
    
    % extract roc curve
    [xAxis, yAxis] = perfcurve(testLabels, modelPredictions, 1);
    
    % plot roc curve
    subplot(2,1,1), plot (xAxis, yAxis), title("ROC Curve"), xlabel("False Positive Rate"), ylabel("True Positive Rate");
    
    % plot table of evaluation metrics
    
    % set table data
    errorRate = 1-accuracy;
    falseAlarmRate = 1-specificity;
    f1 = (2 * sensitivity * precision) / (sensitivity + precision);
    tblHeadings = {"Accuracy", "Error Rate", "Sensitivty", "Precision", "Specifity", "False Alarm Rate", "F-measure", "AUC"};
    tblData = horzcat(accuracy, errorRate, sensitivity, precision, specificity, falseAlarmRate, f1, trapz(xAxis, yAxis));
    
    % set table properties such that it can sit in a subplot
    plotTwo = subplot(2,1,2);
    plotPos = get(plotTwo, "Position");
    plotUnits = get(plotTwo, "Units");
    delete(plotTwo);
    uitable("Data", tblData, "ColumnName", tblHeadings, "Units", plotUnits, "Position", plotPos);
    
end