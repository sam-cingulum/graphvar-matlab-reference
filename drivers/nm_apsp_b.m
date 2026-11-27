function D = nm_apsp_b(M)
% Option (b) for small_world_propensity's graphallshortestpaths(M): BCT 2019 distance_wei on the same
% lengths (0 = no edge in BCT).
L = full(M);
L(isinf(L)) = 0;
n = size(L, 1);
L(1:n + 1:end) = 0;
D = distance_wei(L);
end
