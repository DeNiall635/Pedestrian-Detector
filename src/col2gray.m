%%ideally we want to take the images from the training datasets and convert
%%them to grayscale since object identification becomes much easier once
%%once is removed

%Get the working directories of the datasets
dirTrainPos = strcat(pwd,'\training\pos\');
dirTrainNeg = strcat(pwd,'\training\neg\');

%Specify the directories in which we will store the grayscale images

tImPos = dir(fullfile(dirTrainPos,'*.jpg'));
tImNeg = dir(fullfile(dirTrainNeg,'*.jpg'));

%dirCol2Gray(dirTrainPos);
dirCol2Gray(dirTrainNeg);

