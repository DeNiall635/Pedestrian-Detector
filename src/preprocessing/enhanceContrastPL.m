function Iout = enhanceContrastPL(Iin, t)
%% Automatic power law contrast enhancement, uses a default threshold of 0.5 i.e 50% of pixels
    if (nargin == 1)
        t = 0.5;
    end
    
    tPixel = uint8(255 * t);
    
    [pixels, grayLevels] = imhist(Iin);
    % should be 160 x 96 = 15360
    pixelCount = size(Iin,1) * size(Iin,2);
    % no. of pixels below threshold t
    leT = 0;
    % no. of pixels above threshold t
    greT = 0;
    
    for i = 0:255
       if (i < 50) 
           leT = leT + pixels(i+1);
       elseif (i > 200)
           greT = greT + pixels(i+1);
       end 
    end
    
    if (leT / pixelCount > t)
        gamma = 0.5;
    elseif (greT / pixelCount > t)
        gamma = 2;
    else
        gamma = 1;
    end
    
    Lut = contrast_PL_LUT(gamma);
    Iout = intlut(Iin, Lut);    
end