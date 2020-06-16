% Returns a lookup table for the brightness based on input value c 
%(TYPE - int)
function Lut = brightnessLUT(c)
    Lut = zeros(1,256);
    for i = 1:256
        base = i - 1;
        input = base + c;
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