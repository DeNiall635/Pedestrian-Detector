%% Function for plotting ROC curves separate from evaluation metrics
function auc = plotROC(testLabels, modelPredictions, plotTitle)
    
    if (nargin < 3)
        plotTitle = 'ROC Curve';
    end

    [xAxis, yAxis] = perfcurve(testLabels, modelPredictions, 1);
    % plot roc curve
    plot (xAxis, yAxis), title(plotTitle), xlabel("False Positive Rate"), ylabel("True Positive Rate");

    % return area under curve
    auc = trapz(xAxis, yAxis);
    
end