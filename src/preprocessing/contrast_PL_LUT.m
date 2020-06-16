function Lut = contrast_PL_LUT(gamma)
    Lut = zeros(1,256);
    for i = 1:256
        base = i - 1;
        input = double(base^gamma) / double(255^(gamma-1));
        if input < 0
            Lut(i) = 0;
        elseif input > 255
            Lut(i) = 255;
        else
            Lut(i) = input;
        end
                
    end
    Lut = uint8(Lut);
end