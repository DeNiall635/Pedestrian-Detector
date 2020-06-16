%% USING THE ADABOOST EMSEMBLE TO TRAIN WEAK TREE CLASSIFIERS

clear all;
close all;

load('trainingDataPreprocessed.mat');
load('trainHogFeaturesPreprocessed.mat');

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

% run PCA for dimensionality reduction
[eigenVector, eigenvalues, meanX, Xpca] = PrincipalComponentAnalysis(trHogs);

BoostModel = fitensemble(Xpca, trLabels, 'AdaBoostM1', 100, 'Tree');

% Create a new video file to store the frames, and set fr to 1 such that we
% have time to view every bb in the frame.

video = VideoWriter('models/ppHOG_BOOSTING_DT.avi');
video.FrameRate = 1;
open(video);

pedestrianDetections = [];

modelDetections = [];

% scales for the sliding window
winScales = [0.25, 0.33, 0.5, 0.75,1, 1.25];

% loop over each test image in the test dataset (Mine)
tsDim = [480 640];
tic
total = size(tsImg,1);
for i=1:total
    fprintf('Test Image %i', i)
    
    % get test image at i and reshape to original dimensions.
    tsIm = tsImg(i,:);
    tsIm = reshape(tsIm,tsDim);
    bestPositions = zeros(0,5);

    
    for s = 1:size(winScales,2)
        windowScale = winScales(s);
    % horizontal sliding window
    
        % set sizes for sliding window based on scale
        winH = 160/windowScale;
        winW = 96/windowScale;
    
        positions = zeros(0,5);
        
        imHeight = size(tsIm,1);
        imWidth = size(tsIm,2);
        
        for rowIndex=1:20:imHeight
           for columnIndex=1:20:imWidth
               heightFit = rowIndex+winH-1 <= imHeight;
               widthFit = columnIndex+winW-1 <= imWidth;
               if (heightFit && widthFit)
                    hWindow = tsIm([rowIndex:rowIndex+winH-1],[columnIndex:columnIndex+winW-1]);
                    hWindow = imresize(hWindow, [160 96]);
                    winHog = (hog_feature_vector(hWindow)-meanX)*eigenVector;
                    [label,score] = predict(BoostModel, winHog);
                    if (label == 1)
                        positions = [positions; [score(1), columnIndex, rowIndex, winW, winH]];
                    end
               end
           end
        end

        % vertical sliding window

        for columnIndex = 1:20:imWidth
            for rowIndex = 1:20:imHeight
                widthFit = columnIndex+winW-1 <= imWidth;
                heightFit = rowIndex+winH-1 <= imHeight;
                if (widthFit && heightFit)
                    vWindow = tsIm([rowIndex:rowIndex+winH-1],[columnIndex:columnIndex+winW-1]);
                    vWindow = imresize(vWindow, [160 96]);
                    winHog = (hog_feature_vector(vWindow)-meanX)*eigenVector;
                    [label,score] = predict(BoostModel, winHog);
                    if (label == 1)
                        positions = [positions; [score(1), columnIndex, rowIndex, winW, winH]];
                    end
                end
            end
        end
        
        if (size(positions,1) > 1)
            best = 0;
            bestIndex = 0;    
            for p = 1:size(positions,1)
                posScore = abs(positions(p,1));
                if (posScore > best)
                    best = posScore;
                    bestIndex = p; 
                end
            end
            bestPositions = [bestPositions; positions(bestIndex,2:end)];
        end
        bestPositions = simpleNMS(bestPositions, 0.3);
    end
    
    figure(2)
    imshow(tsIm,[]), title('image with window');
    for bounds = 1:size(bestPositions,1)
        rectangle('Position',[bestPositions(bounds,1),bestPositions(bounds,2),bestPositions(bounds,3),bestPositions(bounds,4)], 'EdgeColor','g','LineWidth',2);
    end
    
    newFrame = getframe(2);
    newFrame = imresize(newFrame.cdata, [500 800]);
    writeVideo(video, newFrame);
    
    totalPositions = size(bestPositions,1);
    imagePredictedDetection.pedCount = totalPositions;
    imagePredictedDetection.positions = bestPositions;
    imagePredictedDetection.isOverlap = zeros(1, totalPositions);
    
    modelDetections = [modelDetections, imagePredictedDetection];
    
end

close(video);
toc

save('HOG_B-DT_DETECTOR','modelDetections');
compareDetections(modelDetections,'HOG_B-DT');
