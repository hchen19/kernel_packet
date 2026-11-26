function x = hc(j, lo, hi)
% Hyperbolic cross (bisection) design of level j: 2^j-1 points in (lo,hi), nested in j
x = lo + (hi - lo) * (1:2^j-1) / 2^j;
end
