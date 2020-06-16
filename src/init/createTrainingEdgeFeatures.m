%% Calculate Edge feature vectors for all training data and store them in the project folder for quick loading in future.

close all;
clear all;

load('trainingDataPreprocessed.mat');

imDim = [160,96];

trEdges = getEdges(trIm, imDim);

save('trainEdgeFeatures','trEdges');
