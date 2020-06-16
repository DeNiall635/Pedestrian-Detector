% Enhance Contrast - Automatic Linear Stretching.
function [Iout, Lut] = enhanceContrastALS(Iin)
    % Double min/max needed for matrix
    % OR use max(Image(:)) & (:) considers the entire matrix
    i1 = min(Iin(:));
    i2 = max(Iin(:));
    m = double(255) / double((i2 - i1));
    c = -(m*i1);
    
    Lut = contrast_LS_LUT(m,c);
    Iout = uint8(double(Iin*m + c));
end