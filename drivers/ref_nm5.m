function ref_nm5(bctDir, gvDir, inDir, outDir)
% Fifth null-model battery (R2026b): Louvain and community detection. Input: inDir/nm5.json (tools/gen_nm5.py).
% Writes outDir/nm5.jsonl. Parts:
%   env    version, which() of the community functions with BCT 2019 first and with GraphVar's ext/BCT first
%   cl     rng(seed); [M, Q] = community_louvain(W, gamma) (BCT 2019 first); after = rand(1, 4). Then rng(seed) again and
%          cl_trace (community_louvain line for line, logging): B0 (B after symmetrising), perm (every pass's randperm, in
%          order), passn (n of each pass), moves (1 where the node moved, per visited node), Qlev (Q at each level, the
%          first is the initial Q), Mt, Qt, after_t. notes.same_as_bct = M, Q and after equal to the untraced call.
%   clgv   GraphVar's ext/BCT community_louvain(W) first on the path (default objective 'negative_asym'): M, Q, after
%   old    GraphVar's ext/BCT modularity_louvain_und(W): Ci, Q, after
%   items  a GraphVar menu helper fn(W) with gui/helper on the path and ext/BCT first ("gv") or after BCT 2019 ("bct"):
%          out1 (and out2 for the two-output cOut helpers), after; or the MATLAB error
%   prim   X (n x n) from rng(seed) as gen_nm4 sum3: trace(X) (kind trace) or, with L = rand(n) < 0.5 drawn next,
%          sum(X(L)) and nnz(L) (kind lsum)
q = jsondecode(fileread(fullfile(inDir, 'nm5.json')));
if ~isfolder(outDir), mkdir(outDir); end
out = fullfile(outDir, 'nm5.jsonl');
if isfile(out), delete(out); end
src = fullfile(gvDir, 'src');
gvBCT = fullfile(src, 'ext', 'BCT');
addpath(fullfile(src, 'gui', 'helper'));
addpath(bctDir);
fns = {'community_louvain', 'modularity_louvain_und', 'agreement', 'consensus_und', 'modularity_und', ...
       'diversity_coef_sign', 'dummyvar', 'modularity_louvain_QOut_und', 'modularity_louvain_cOut_und', 'hpos_Louvain', ...
       'modularity_cOut_und', 'hpos_Newman'};

rec = newrec('nm5-env', 'env');
rec.version = version;
for f = fns, rec.notes.(['which_bct_' f{1}]) = clean(which(f{1}), bctDir, gvDir); end
addpath(gvBCT);
for f = fns, rec.notes.(['which_gv_' f{1}]) = clean(which(f{1}), bctDir, gvDir); end
rmpath(gvBCT);
emit(out, rec);

G = struct();
for g = cellify(q.graphs)
    g = g{1};
    G.(key(g.name)) = reshape(hex2num(cellstr(g.W)), g.n, g.n);
end

for c = cellify(q.cl)
    c = c{1};
    rec = newrec(c.case, 'cl');
    rec.graph = c.graph; rec.gamma = c.gamma; rec.seed = c.seed;
    W = G.(key(c.graph));
    try
        rng(c.seed, 'twister');
        [M, Q] = community_louvain(W, c.gamma);
        after = rand(1, 4);
        rec.outputs.M = M; rec.outputs.Q = Q; rec.outputs.after = after;
        rng(c.seed, 'twister');
        [Mt, Qt, tr] = cl_trace(W, c.gamma);
        after_t = rand(1, 4);
        f = fieldnames(tr);
        for i = 1:numel(f), rec.outputs.(f{i}) = tr.(f{i}); end
        rec.outputs.Mt = Mt; rec.outputs.Qt = Qt; rec.outputs.after_t = after_t;
        rec.notes.same_as_bct = double(isequal(M, Mt) && isequaln(Q, Qt) && isequal(after, after_t));
    catch err
        rec.error = struct('identifier', err.identifier, 'message', err.message);
    end
    emit(out, rec);
end

addpath(gvBCT);
for c = cellify(q.clgv)
    c = c{1};
    rec = newrec(c.case, 'clgv');
    rec.graph = c.graph; rec.seed = c.seed;
    W = G.(key(c.graph));
    try
        rng(c.seed, 'twister');
        [rec.outputs.M, rec.outputs.Q] = community_louvain(W);
        rec.outputs.after = rand(1, 4);
    catch err
        rec.error = struct('identifier', err.identifier, 'message', err.message);
    end
    emit(out, rec);
end
for c = cellify(q.old)
    c = c{1};
    rec = newrec(c.case, 'old');
    rec.graph = c.graph; rec.seed = c.seed;
    W = G.(key(c.graph));
    try
        rng(c.seed, 'twister');
        [rec.outputs.Ci, rec.outputs.Q] = modularity_louvain_und(W);
        rec.outputs.after = rand(1, 4);
    catch err
        rec.error = struct('identifier', err.identifier, 'message', err.message);
    end
    emit(out, rec);
end
rmpath(gvBCT);

two = {'modularity_louvain_cOut_und', 'modularity_cOut_und'};
for c = cellify(q.items)
    c = c{1};
    rec = newrec(c.case, 'items');
    rec.fn = c.fn; rec.graph = c.graph; rec.seed = c.seed; rec.order = c.order;
    W = G.(key(c.graph));
    if strcmp(c.order, 'gv'), addpath(gvBCT); else, addpath(gvBCT, '-end'); end
    rec.notes.which_community_louvain = clean(which('community_louvain'), bctDir, gvDir);
    rec.notes.which_agreement = clean(which('agreement'), bctDir, gvDir);
    try
        rng(c.seed, 'twister');
        if any(strcmp(c.fn, two))
            [o1, o2] = feval(c.fn, W);
            rec.outputs.out1 = o1; rec.outputs.out2 = o2;
        else
            rec.outputs.out1 = feval(c.fn, W);
        end
        rec.outputs.after = rand(1, 4);
    catch err
        rec.error = struct('identifier', err.identifier, 'message', err.message);
    end
    rmpath(gvBCT);
    emit(out, rec);
end

for c = cellify(q.prim)
    c = c{1};
    rec = newrec(c.case, 'prim');
    rec.kind = c.kind; rec.n = c.n; rec.seed = c.seed;
    try
        rng(c.seed, 'twister');
        U1 = rand(c.n); U2 = rand(c.n);
        X = (U1 - 0.5) .* 2.^(floor(64 * U2) - 32);
        if strcmp(c.kind, 'trace')
            rec.outputs.t = trace(X);
        else
            L = rand(c.n) < 0.5;
            rec.outputs.s = sum(X(L));
            rec.outputs.nnz = nnz(L);
        end
    catch err
        rec.error = struct('identifier', err.identifier, 'message', err.message);
    end
    emit(out, rec);
end
end

function [M, Q, tr] = cl_trace(W, gamma)
% BCT 2019 community_louvain(W, gamma) with M0 and B absent ('modularity'), line for line, logging.
W = double(W);
n = length(W);
s = sum(sum(W));
if min(min(W)) < -1e-10
    error('nm5:negative', 'negative weights');
end
B = (W - gamma * (sum(W, 2) * sum(W, 1)) / s) / s;
M0 = 1:n;
[~, ~, Mb] = unique(M0);
M = Mb;
B = (B + B.') / 2;
tr.B0 = B;
tr.perm = []; tr.passn = []; tr.moves = []; tr.Qlev = [];
Hnm = zeros(n, n);
for m = 1:max(Mb)
    Hnm(:, m) = sum(B(:, Mb == m), 2);
end
Q0 = -inf;
Q = sum(B(bsxfun(@eq, M0, M0.')));
tr.Qlev(end + 1) = Q;
first_iteration = true;
while Q - Q0 > 1e-10
    flag = true;
    while flag
        flag = false;
        p = randperm(n);
        tr.perm = [tr.perm, p]; tr.passn(end + 1) = n;
        for u = p
            ma = Mb(u);
            dQ = Hnm(u, :) - Hnm(u, ma) + B(u, u);
            dQ(ma) = 0;
            [max_dQ, mb] = max(dQ);
            if max_dQ > 1e-10
                flag = true;
                Mb(u) = mb;
                Hnm(:, mb) = Hnm(:, mb) + B(:, u);
                Hnm(:, ma) = Hnm(:, ma) - B(:, u);
                tr.moves(end + 1) = 1;
            else
                tr.moves(end + 1) = 0;
            end
        end
    end
    [~, ~, Mb] = unique(Mb);
    M0 = M;
    if first_iteration
        M = Mb;
        first_iteration = false;
    else
        for u = 1:n
            M(M0 == u) = Mb(u);
        end
    end
    n = max(Mb);
    B1 = zeros(n);
    for u = 1:n
        for v = u:n
            bm = sum(sum(B(Mb == u, Mb == v)));
            B1(u, v) = bm;
            B1(v, u) = bm;
        end
    end
    B = B1;
    Mb = 1:n;
    Hnm = B;
    Q0 = Q;
    Q = trace(B);
    tr.Qlev(end + 1) = Q;
end
end

function s = clean(w, bctDir, gvDir)
s = strrep(strrep(w, bctDir, '<bct>'), gvDir, '<gv>');
end

function k = key(name)
k = ['g_' strrep(strrep(name, '-', '_'), '.', 'p')];
end

function c = cellify(s)
if iscell(s), c = s(:)'; else, c = num2cell(s(:)'); end
end

function rec = newrec(caseId, part)
rec = struct('case', caseId, 'part', part, 'outputs', struct(), 'notes', struct());
end

function emit(out, rec)
f = fieldnames(rec.outputs);
for i = 1:numel(f)
    v = rec.outputs.(f{i});
    if ischar(v) || isstruct(v) || iscell(v), continue, end
    v = double(v);
    if isempty(v)
        rec.outputs.(f{i}) = struct('size', size(v), 'hex', {{}});
    else
        rec.outputs.(f{i}) = struct('size', size(v), 'hex', {cellstr(num2hex(v(:)))'});
    end
end
fid = fopen(out, 'a');
fprintf(fid, '%s\n', jsonencode(rec));
fclose(fid);
end
