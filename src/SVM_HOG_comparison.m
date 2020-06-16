vid1 = VideoReader('HOG_SVM.avi');
vid1Frames = read(vid1);
vid2 = VideoReader('ppHOG_SVM.avi');
vid2Frames = read(vid2);
vid3 = VideoReader('ppHOG_SVM_NMS.avi');
vid3Frames = read(vid3);

nFrame = 100;

for i = 1:nFrame
   frame1 = vid1Frames(:,:,i);
   frame2 = vid2Frames(:,:,i);
   frame3 = vid3Frames(:,:,i);
   
   subplot(1,3,1), imshow(frame1), title('w/o pp & ssw');
   subplot(1,3,2), imshow(frame2), title('w/ pp & ssw');
   subplot(1,3,3), imshow(frame3), title('w/ pp, ssw & nms');
   
   pause(3);
end