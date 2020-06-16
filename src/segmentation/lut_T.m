function Lut = lut_T(t)
    Lut = zeros(1,256);
    
    for i = 1:256
        if (i-1) < t
            Lut(i) = 255;
        else
            Lut(i) = 0;
        end
    end
    Lut = uint8(Lut);
    
end