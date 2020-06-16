%We want to load in the images from the pedestrian folder and create a video from them
%pedDir = 'C:\Users\callu\Documents\matlab_madness-csc3061_assignment\src\test\pedestrian\';
pedDir = strcat(pwd,'\test\pedestrian');
imgFiles = dir(fullfile(pedDir,'*.jpg'));

pedColVid = VideoWriter(strcat(testDir,'pedColVid.avi'));
open(pedColVid);
for i = 1:length(imgFiles)
   imFile = fullfile(pedDir,imgFiles(i).name);
   im = imread(imFile);
   
   frame = im2frame(im);
   writeVideo(pedColVid,frame);
   pause(0.01);
end
close(pedColVid);
%at this point, the avi is made from the images
