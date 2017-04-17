function xPower=PowerNormalization(x, alpha)
%Compute power normalization for a given vector/matrix

if (nargin<2)
   alpha =0.5;         
end

xPower=sign(x).*(abs(x).^alpha);