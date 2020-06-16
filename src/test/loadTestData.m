%% script to load the position data from the test.cdataset
% need to parse the image name, actual pedestrian count and the bounding
% positions of the pedestrians

clear all;
close all;

testFile = fopen('test.cdataset');

fileName = fgetl(testFile);
imageCount = uint8(str2double(fgetl(testFile)));

testPedData = [];

fileCount = 1;
while fileCount <= imageCount
    
    newLine = fgetl(testFile);
    
    if (ischar(newLine))
        
        newLineSplit = strsplit(newLine);
        testImageName = newLineSplit{1};
        testImagePeds = uint8(str2double(newLineSplit{2}));
        
        pedPositions = [];
        for i = 3:length(newLineSplit)-1
            newStr = newLineSplit{i};
            if strcmp(newStr, '0') == 0
                pedPositions = [pedPositions; str2double(newStr)];
            end
        end
        pedPositions = reshape(pedPositions, 4, testImagePeds).';
        
        testImageData.name = testImageName;
        testImageData.pedCount = testImagePeds;
        testImageData.positions = pedPositions;
        
        testPedData = [testPedData; testImageData];
    end
    
    fileCount = fileCount + 1;
end

save('testPedestrianPositions','testPedData');

fclose(testFile);
