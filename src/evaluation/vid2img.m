vid2img('reportData/modelComparisons/HOG_B-DT.avi', 'HOG_B-DT frames/');

function images = vid2img(videoName, storeDir)
    video = VideoReader(videoName);
    video = read(video); 
    
    noFrames = size(video,4)
    
    for f = 1:noFrames
        
        vidFrame = video(:,:,:,f);
        imshow(vidFrame)
        imName = sprintf('frame_%i', f);
        imwrite(vidFrame,strcat('reportData/modelComparisons/', storeDir, imName, '.jpg'));
        
    end
    
end