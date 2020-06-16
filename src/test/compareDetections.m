function compareDetections(modelDetections, resultFileName)

clf

% load pre-computed pedestrian positions
load('testPedestrianPositions.mat');

% conver the structs into cells for easier operations
testPedData = struct2cell(testPedData);
modelDetections = struct2cell(modelDetections);

% initialise the evaluation metrics
imageNames = [];
accuracies = [];
correctDetections = [];
actualDetections = [];
errorRates = [];
truePositives = [];
falsePositives = [];
falseNegatives = [];

comparisonVideo = VideoWriter(strcat('reportData/modelComparisons/', resultFileName, '.avi'));
comparisonVideo.FrameRate = 1;
open(comparisonVideo);

% we check only the images where the model made a detection
for i = 1:size(modelDetections,3)
    
    % image name
    imName = testPedData{1,i}
    testImage = imread(strcat('test/',imName));
    
    % pedestrians in the image
    testPedCount = testPedData{2,i};
    
    % bb positions of the pedestrians
    testPedPositions = testPedData{3,i};
    
    % detected pedestrians
    detectedPedCount = modelDetections{1,i};
    
    % bb positions of detected pedestrians
    detectedPedPositions = modelDetections{2,i};
    
    % used to check for overlapping bbs over test images
    detectedIsOverlap = modelDetections{3,i};
    
    % number of successful detections
    correctDetection = 0;
    
    for j = 1:testPedCount
        
        % get the actual pedestrian position and its bb area
        pedPosition = testPedPositions(j,:);
        pedPositionArea = pedPosition(3) * pedPosition(4);
        
        % now check all detected positions for an intersection
        for k = 1:size(detectedPedPositions,1)
            
            detectedPosition = detectedPedPositions(k,:);
            
            positionIntersect = rectint(pedPosition, detectedPosition);
            
            % if intersection is greater than a threshold, mark it as a
            % correct detection, if there isn't an overlapping bb that
            % has already detected it.
            if (positionIntersect / pedPositionArea > 0.5)
                
                if (detectedIsOverlap(1,k) == 0)
                    
                    correctDetection = correctDetection + 1;
                    detectedIsOverlap(1,k) = 1;
                    
                end
                
            end
            
        end
        
    end
    
    figure(1)
    % display the current test image
    imshow(testImage);
    
    % draw the actual pedestrian bb in green
    for b = 1:testPedCount
        rectangle('Position', [testPedPositions(b, 1)-(testPedPositions(b, 3)/2) ...
            testPedPositions(b, 2)-(testPedPositions(b, 4)/2) ...
            testPedPositions(b, 3) ...
            testPedPositions(b, 4)] ,...
            'EdgeColor', 'g', 'LineWidth', 2);
    end
    
    % draw our detected bb in red
    for b = 1:detectedPedCount
        rectangle('Position', [detectedPedPositions(b, 1)-(detectedPedPositions(b, 3)/2) ...
            detectedPedPositions(b, 2)-(detectedPedPositions(b, 4)/2) ...
            detectedPedPositions(b, 3) ...
            detectedPedPositions(b, 4)] ,...
            'EdgeColor', 'r', 'LineWidth', 2);
    end
    
    % write the figure to a new frame
    newFrame = getframe(1);
    newFrame = imresize(newFrame.cdata, [500 800]);
    writeVideo(comparisonVideo, newFrame);
    
    % calculate the evaluation metrics for this frame
    modelAccuracy = round(double(correctDetection) / double(testPedCount),3);
    
    errorRate = 1 - modelAccuracy;
    
    truePositive = correctDetection;
    falsePositive = detectedPedCount - correctDetection;
    
    falseNegative = testPedCount - correctDetection;
    
    % update the metrics
    imageNames = [imageNames; testPedData{1,i}];
    accuracies = [accuracies; modelAccuracy];
    correctDetections = [correctDetections; correctDetection];
    actualDetections = [actualDetections; testPedCount];
    errorRates = [errorRates; errorRate];
    truePositives = [truePositives; truePositive];
    falsePositives = [falsePositives; falsePositive];
    falseNegatives = [falseNegatives; falseNegative];
    
    
end

close(comparisonVideo);

% % set table data
% tblHeadings = {"Image", "Accuracy", "Error Rate", "Correct Detections", "Actual Detections","TP", "FP", "FN"};
% tblData = horzcat(imageNames, accuracies, errorRates, correctDetections, actualDetections, truePositives, falsePositives, falseNegatives);
% uitable("Data", tblData, "ColumnName", tblHeadings);
% %disp(tblData);

% save the table into a csv .txt
tbl = table(imageNames, accuracies, errorRates, correctDetections, actualDetections, truePositives, falsePositives, falseNegatives);
tbl.Properties.VariableNames = {'Image', 'Accuracy', 'Error Rate', 'Correct Detections', 'Actual Detections', 'TP', 'FP', 'FN'};
writetable(tbl,strcat('reportData/modelComparisons/', resultFileName),'Delimiter',',', 'WriteRowNames', true);

end