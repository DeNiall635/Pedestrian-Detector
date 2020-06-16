function Iout = enhanceContrastLS(Iin, m, c)
    Lut = contrast_LS_LUT(m,c);
    Iout = intlut(Iin, Lut);
end