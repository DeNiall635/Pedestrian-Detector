%% Calculate edges for a given set of image data using Sobel algorithm
function imEdges = getEdges(images, imDim)
    imEdges = [];
    
    for i = 1:size(images,1)
       % reshape image data to original format
       newIm = images(i,:);
       newIm = reshape(newIm, imDim);
       imEdge = edge(newIm, 'Sobel');
       imEdge = reshape(imEdge, 1, size(imEdge, 1) * size(imEdge, 2));
       imEdge = double(imEdge);
       imEdges = [imEdges; imEdge];
       
    end
    
end