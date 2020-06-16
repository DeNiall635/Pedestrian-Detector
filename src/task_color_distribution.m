%Get the working directories of the datasets
dirTrainPos = strcat(pwd,'\training\pos\');
dirTrainNeg = strcat(pwd,'\training\neg\');

filesPos = dir(fullfile(dirTrainPos,'*.jpg'));

dirLength = length(filesPos);

redVals = zeros(dirLength,1);
greenVals = zeros(dirLength,1);
blueVals = zeros(dirLength,1);


for i = 1:dirLength
    img = imread(fullfile(dirTrainPos,filesPos(i).name));
    subplot(1,4,1), imshow(img);
    % Red channel
    red = img(:,:,1);
    %subplot(1,4,2), imshow(red);
    redVals(i,1) = uint8(mean(red(:)));
    % Green channel
    green = img(:,:,3);
    %subplot(1,4,4), imshow(green);
    greenVals(i,1) = uint8(mean(green(:)));
    % Blue channel
    blue = img(:,:,2);
    %subplot(1,4,3), imshow(blue);
    blueVals(i,1) = uint8(mean(blue(:)));
end

subplot(1,3,1), histogram(redVals), title("Red Channel");
subplot(1,3,2), histogram(greenVals), title("Green Channel");
subplot(1,3,3), histogram(blueVals), title("Blue Channel");