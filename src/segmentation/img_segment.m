imTest = imread('training/pos_grayscale/gray_img00000.jpg');
imMean = mean(imTest(:));
imSd = std(double(imTest(:)));
a = 0.005;
t = uint8(imMean + a*imSd);

lut = lut_T(t);
binIm = intlut(imTest,lut);
subplot(2,3,1), imshow(imTest), title('original');
subplot(2,3,4), histogram(imTest), title('histogram');
subplot(2,3,2), imshow(binIm), title('binary');

