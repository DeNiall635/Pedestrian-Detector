%% Test different preprocessing techniques on training data to see which
%% will enhance it most

clear all;
close all;

load('trainingData.mat');

% we will randomly pick 3 images from the training data and apply the
% techniques to them, then save them for future reference

r = randi(size(trIm,1), 5);
rSize = length(r);

preprocessed = [];

for i=1:rSize
    im = trIm(r(i),:);
    rIm = uint8(reshape(im, [160 96]));
    rImO = rIm;
    
    
    % show original image
    figure(1)
    subplot(2,3,1), imshow(rIm), title('1');
    
    % Contrast enhance (automatic)
    rIm = enhanceContrastALS(rIm);
    subplot(2,3,2), imshow(rIm), title('2');
    
    % Histogram Equalisation
    rIm = histeq(rIm,256);
    subplot(2,3,4), imshow(rIm), title('3');
    
    % Power Law enhance
    rIm = enhanceContrastPL(rIm);
    subplot(2,3,5), imshow(rIm), title('4');
    
    % plot original and final histograms
    subplot(2,3,3), histogram(rImO), xlabel('intensity'), ylabel('frequency'), title('Original')
    subplot(2,3,6), histogram(rIm), xlabel('intensity'), ylabel('frequency'), title('Final')
    
    imFile = sprintf('reportData/preprocessing/preprocess_img_%i.png', r(i));
    saveas(gcf, imFile);
    
end

% for i = 1:size(trIm,1)
%    im = trIm(i,:);
%    rIm = uint8(reshape(im, [160 96]));
%
%     % Combine all preprocessing methods into one
%     fIm = enhanceContrastALS(uint8(rIm));
%     fIm = histeq(fIm);
%     fIm = enhanceContrastPL(uint8(fIm));
%
%     % Take preprocessed image and reshape, store in original dataset.
%     fImR = reshape(fIm, 1, size(fIm,1) * size(fIm,2));
%     fImR = double(fImR);
%     trIm(i,:) = fImR;
% end
%
% save('trainingDataPreprocessed','trIm','trLabels');
