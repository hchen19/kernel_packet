function C = matern(X1, X2, k, rho)
% Product Matern-(k-2)/2 kernel matrix between the rows of X1 (m*d) and X2 (n*d) with lengthscales rho,
% accumulated one dimension at a time: K(r) = P(z) exp(-z), z = sqrt(k-2)*|r|/rho
p = (k-3)/2;
a = ones(1, p+1);
for j = 1:p, a(j+1) = a(j) * 2*(p-j+1) / ((2*p-j+1) * j); end
d = size(X1, 2);
c = sqrt(k-2) ./ (rho .* ones(1, d));
P = 1;
S = 0;
for i = 1:d
    Z = abs(X1(:,i) - X2(:,i)') * c(i);
    Pi = a(end);
    for j = p:-1:1, Pi = Pi .* Z + a(j); end
    P = P .* Pi;
    S = S + Z;
end
C = P .* exp(-S);
end
