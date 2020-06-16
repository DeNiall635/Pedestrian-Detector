clear all;
close all;

load('trainingData.mat');
load('trainHogFeatures.mat');

load('testData.mat');

imDim = [160 96];

% Display 3 random hogs and their associated training images.

rIndexes = randi(size(trIm,1), [1 3]);
figure(1)
for i = 1:length(rIndexes)
    % offset for subplotting
    offset = ((i-1)*2);
    % get index from random index array
    index = rIndexes(i);
    % reshape training image to original dimensions.
    rIm = reshape(trIm(index,:), imDim);
    subplot(length(rIndexes),2,1 + offset), imshow(rIm,[]), title(sprintf('image %i',index));
    subplot(length(rIndexes),2,2 + offset), showHog(trHogs(index,:),imDim), title(sprintf('image %i hog', index));
end

numNeigh = 3;
knnModel = fitcknn(trHogs,trLabels, 'NumNeighbours',numNeigh);

modelDetections = [];

% Create a new video file to store the frames, and set fr to 1 such that we
% have time to view every bb in the frame.

<<<<<<< HEAD
video = VideoWriter('C:\Users\callu\Desktop\KNN Video\knn_detector.avi');
=======
video = VideoWriter('models/HOG_KNN.avi');
>>>>>>> e86c24581501c50f6eec0332829eea660cf34c52
video.FrameRate = 1;
open(video);

pedestrianDetections = [];

tsDim = [480 640];
tic
for i=1:size(tsImg,1)
    % get test image at i and reshape to original dimensions.
    tsIm = tsImg(i,:);
    tsIm = reshape(tsIm,tsDim);
    bestPositions = zeros(0,5);
    positions = zeros(0,5);
    
    % horizontal sliding window
    
    for rowIndex=1:40:size(tsIm, 1)
       for columnIndex=1:40:size(tsIm,2)
           heightFit = rowIndex+160-1 <= size(tsIm,1);
           widthFit = columnIndex+96-1 <= size(tsIm,2);
           if (heightFit && widthFit)
                hWindow = tsIm([rowIndex:rowIndex+160-1],[columnIndex:columnIndex+96-1]);
                winHog = hog_feature_vector(hWindow);
                [label,score] = predict(knnModel, winHog);
                if (label == 1)
                    positions = [positions; [score(1), columnIndex, rowIndex, (columnIndex+96-1), (rowIndex+96-1)]];
                end
           end
       end
    end
    
    % vertical sliding window
    
    for columnIndex = 1:40:size(tsIm, 2)
        for rowIndex = 1:40:size(tsIm, 1)
            widthFit = columnIndex+96-1 <= size(tsIm, 2);
            heightFit = rowIndex+160-1 <= size(tsIm, 1);
            if (widthFit && heightFit)
                vWindow = tsIm([rowIndex:rowIndex+160-1],[columnIndex:columnIndex+96-1]);
                winHog = hog_feature_vector(vWindow);
                [label,score] = predict(knnModel, winHog);
                if (label == 1)
                    positions = [positions; [score(1), columnIndex, rowIndex, (columnIndex+96-1), (rowIndex+96-1)]];
                end
            end
        end
    end
    
    best = 0;
    bestIndex = 0;    
    for p = 1:size(positions,1)
        posScore = abs(positions(p,1));
        if (posScore > best)
            best = posScore;
            bestIndex = p; 
        end
    end
    bestPositions = [bestPositions;positions(bestIndex,2:end)];
    
    figure(2)
    imshow(tsIm,[]), title('image with window');
    for bounds = 1:size(bestPositions,1)
        rectangle('Position',[bestPositions(bounds,1),bestPositions(bounds,2),bestPositions(bounds,3),bestPositions(bounds,4)], 'EdgeColor','g','LineWidth',2);
    end
    
    newFrame = getframe(2);
    newFrame = imresize(newFrame.cdata, [800 500]);
    writeVideo(video, newFrame);
    
    totalPositions = size(bestPositions,1);
    imagePredictedDetection.pedCount = totalPositions;
    imagePredictedDetection.positions = bestPositions;
    imagePredictedDetection.isOverlap = zeros(1, totalPositions);
    
    modelDetections = [modelDetections, imagePredictedDetection];
end

close(video);
toc
save('HOG_KNN_DETECTOR','modelDetections'); %not necessary but save the detections for each image
compareDetections(modelDetections,'HOG_KNN');