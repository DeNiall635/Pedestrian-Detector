%% Calculate HOG feature vectors for all training data and store them in the project folder for quick loading in future.

close all;
clear all;

load('trainingDataPreprocessed.mat');

imDim = [160,96];
%trIm = preprocessed;
trHogs = [];
tic
for i = 1:size(trIm,1)
    rIm = reshape(trIm(i,:), imDim);
    imHog = hog_feature_vector(rIm);
    trHogs = [trHogs; imHog];
end
toc
save('trainHogFeaturesPreprocessed','trHogs');
