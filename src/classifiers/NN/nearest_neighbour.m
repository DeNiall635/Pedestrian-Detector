clear all
close all

% Nearest Neighbour classification

% Load image details into a table

pos_matrix = dir(strcat(pwd, '\src\training\pos_grayscale\*.jpg'));
neg_matrix = dir(strcat(pwd, '\src\training\neg_grayscale\*.jpg'));

test_matrix = dir(strcat(pwd, '\src\test\pedestrian\*.jpg'));

%Initialise positive images' matrix
pos_image_data = [];

for i = 1 : 100
    
	% filename: the name of each individual file in the folder
    filename = strcat(pwd, '\src\training\neg_grayscale\',neg_matrix(i).name);
    % Reading the file
    image = imread(filename);
    data = [];
    [row, col] = size(image);
    for j = 1 : row
        image_row = image(j,:);
        data = [data, image_row];
    end

    neg_image_data = [neg_image_data; data];
    
end

%Initialise negative images' matrix
neg_image_data = [];

for i = 1 : 100

     % filename: the name of each individual file in the folder
    filename = strcat(pwd, '\src\training\neg_grayscale\',neg_matrix(i).name);
    % Reading the file
    image = imread(filename);
    data = [];
    [row, col] = size(image);
    for j = 1 : row
        image_row = image(j,:);
        data = [data, image_row];
    end

    neg_image_data = [neg_image_data; data];
    
end

%Supervised training function that takes the examples and infers a model
model = NNmodel(pos_image_data, neg_image_data);

% Working with negative image data and positive image data
% Load the Test Image 480x640

% IMPORTANT NOTE: This code is used for testing purposes and analyses one
% image only to check if knn is a better feature detector.
noOfDetections = 0;
for i = 1:100 
test_image = strcat(pwd, '\src\test\pedestrian\grayscale\gray_',test_matrix(i).name);

test_image = imread(test_image);

data_test = [];
test_image_data = [];

[r_test, c_test] = size(test_image);
    for j = 1 : r_test
        image_row = test_image(j,:);
        test_data = [data_test, image_row];
    end

test_image_data = [test_image_data; test_data];

% Create an algorithm that splits the "test_image" into images of
% dimensions 160x96 and store the image data in a row. Calculate the
% Eucledian Distance and compare. 

%% Sliding Window

% Scales that we use to divide the test image
scales = 1;   % 54 windows
outputImage = zeros(480, 640);

window_data = [];

window_coordinates = [];

testImage = test_image;
% Reshape to create window
testImage = reshape(testImage, [480 640]);
[rows, columns] = size(testImage);

for s=1:size(scales, 2)
    scale = scales(s);
    windowHeight = 160/scale;
    windowWidth = 96/scale;
    
    for r=90:75:rows
        for c=1:75:columns
            if r+windowHeight <= rows && c+windowWidth <= columns
                window = testImage([r:r+windowHeight], [c:c+windowWidth]);
                c_n = c+windowWidth;
                r_n = r+windowHeight;
                coordinates = [c, r, (c+windowWidth-1)/1.5, (r+windowHeight-1)/1.5];
                figure(1);
                subplot(1,2,1);
                imshow(window);
                
                pause(0.2);
                
                window = imresize(window, [160 96]);
                window_data = NNclassifier(window, model, 3);
                    
                    %If pedestrian is detected
                    if window_data == 1
                        
                        noOfDetections = noOfDetections + 1;
                        
                        window_coordinates = [window_coordinates; coordinates];
                        
                        disp('Pedestrian Detected')
                        disp(noOfDetections)
                        
                        r = size(window_coordinates,1);
                        subplot(1,2,2);
                        imshow(test_image);
                        
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