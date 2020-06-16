%% Calculate HOGs for a given dataset, using the dimensions imDim
function hogs = getHogs(imageDataset, imDim)
    hogs = [];
    for i = 1:size(imageDataset,1)
        rIm = reshape(imageDataset(i,:), imDim);
        rImHog = hog_feature_vector(rIm);
        hogs = [hogs; rImHog];
    end
end