function [imageData,labels] = createDatabaseFromDirectory(directory,defaultLabel)
%CREATEDATABASEFROMDIRECTORY Returns a set of image data and their
%associated labelling {-1 |-> neg, 0 |-> na, 1 |-> pos}
%   directory: string - the directory of the image data
%   defaultLabel: int - the label which all images in the directory will be
%   labelled.

    imageData = [];
    labels = [];
    
    % get files in directory
    imgFiles = dir(fullfile(directory,'*.jpg'));
    
    for i = 1:length(imgFiles)
        % load image
        newImage = imread(imgFiles(i).name);
        
        % check if image data is rgb and convert to grayscale if so
        if size(newImage, 3) > 1 %i.e. not grayscale
            newImage = rgb2gray(newImage);
        end
        
        % convert image data to vector of type double
        imVec = reshape(newImage, 1, size(newImage, 1) * size(newImage, 2));
        imVec = double(imVec);
        
        imageData = [imageData; imVec];
        labels = [labels; defaultLabel];
        
    end
    
end

