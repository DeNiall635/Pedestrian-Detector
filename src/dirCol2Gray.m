% Takes in a directory and converts images in the directory to grayscale
function dirCol2Gray(directoryName)
    %Takes the directory name, creates a directory
    %within that will contain the converted images
    %dir name should be directory with gray_ prefix
    dirGray = fullfile(directoryName, 'grayscale\');
    if exist(dirGray, 'dir')
        disp('Folder already exists. Aborting function.');
        return
    else
        disp('Folder does not exist. Will create one.');
        mkdir(dirGray);
        dirIm = dir(fullfile(directoryName, '*.jpg'));

        for i = 1:length(dirIm)            
            %Load the color image
            imName = dirIm(i).name;
            imFile = fullfile(directoryName, imName);
            imCol = imread(imFile);
            imGray = imCol;
            targetDir = strcat(dirGray, 'gray_', imName);
            if (size(imCol,3) == 3)
                imGray = rgb2gray(imCol);
            end
            imwrite(imGray, targetDir);
        end        
    end

end