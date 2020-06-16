
clear all
close all

load ('trainHogFeaturesPreprocessed.mat');
load ('trainingDataPreprocessed.mat');

% K-Nearest Neighbour classifier
%
% % Load pos, neg and test images and their details into a table for easier
% % access to their file names
pos_matrix = dir(strcat(pwd, '\src\training\pos_grayscale\*.jpg'));
neg_matrix = dir(strcat(pwd, '\src\training\neg_grayscale\*.jpg'));
test_matrix = dir(strcat(pwd, '\src\test\pedestrian\*.jpg'));


%KNN model takes training data labels (imported above) as input and creates a model to be used in
%the next section (test)
%put our dataset in here
%model = KNNmodel(pos_image_data, neg_image_data);
model = fitcknn(trHogs, trLabels, 'NumNeighbours',3);
% working with negative image data and positive image data
% Load the Test Image 480x640
noOfDetections = 0;

for i = 1:numel(test_matrix)
    current_image = strcat(pwd, '\src\test\pedestrian\grayscale\gray_',test_matrix(i).name);
    current_image = imread(current_image);
    
    data_test = [];
    test_image_data = [];
    
    [row_test, col_test] = size(current_image);
    for j = 1 : row_test
        image_row = current_image(j,:);
        test_data = [data_test, image_row];
    end
    
    test_image_data = [test_image_data; test_data];
    
    % Wish to compare the full test image (480x640) multiple times with training images (160x96)
    % Each frame must be split into segments that are 160x96 each
    % Create an algorithm that splits the "test_image" into images of
    % dimensions 160x96 and store the image data in a row.
    % Calculate the Eucledian Distance and compare.
    
    % Sliding Window
    
    % Scales that we use to divide the test image into windows
    scales = 1;   % 54 windows
    outputImage = zeros(480, 640);
    
    window_data = [];
    
    window_coordinates = [];
    
    windowImage = current_image;
    % Reshape to create window
    windowImage = reshape(windowImage, [480 640]);
    [rows, columns] = size(windowImage);
    imshow(windowImage);
    for s=1:size(scales, 2)
        scale = scales(s);
        windowHeight = 160/scale;
        windowWidth = 96/scale;
        
        for r=1:rows
            for c=1:columns
                if r+windowHeight <= rows && c+windowWidth <= columns
                    window = windowImage([r:r+windowHeight], [c:c+windowWidth]);
                    c_n = c+windowWidth;
                    r_n = r+windowHeight;
                    coordinates = [c, r, (c+windowWidth-1)/1.5, (r+windowHeight-1)/1.5];
                    figure(1);
                    subplot(1,2,1);
                    imshow(window);
                    
                    %pause(0.2);
                    
                    window = imresize(window, [160 96]);
                    window_data = KNNclassifier(window, model, 3);
                    
                    %If pedestrian is detected
                    if window_data == 1
                        
                        noOfDetections = noOfDetections + 1;
                        
                        window_coordinates = [window_coordinates; coordinates];
                        
                        disp('Pedestrian Detected')
                        disp(noOfDetections)
                        
                        r = size(window_coordinates,1);
                        subplot(1,2,2);
                        imshow(current_image);
                        
                        hold on;
                        
                        for location=1:r
                            data = window_coordinates(location,:);
                            rectangle('Position',data,'LineWidth',2,'LineStyle','-','EdgeColor','g')
                        end
                        
                    end
                end
            end
        end
    end
end
%Elapsed time is 174.525941 seconds.