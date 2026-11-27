function D = nm_apsp_a(M)
% Option (a) for small_world_propensity's graphallshortestpaths(M): MATLAB's graph/distances on the same
% lengths. M = sparse(1./W) as GraphVar builds it: Inf (from W = 0) means no edge; the diagonal is ignored.
L = full(M);
L(isinf(L)) = 0;
n = size(L, 1);
L(1:n + 1:end) = 0;
D = distances(graph(L));
end
