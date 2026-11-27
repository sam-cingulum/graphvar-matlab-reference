function ref_nm4(bctDir, gvDir, inDir, outDir)
% Fourth null-model battery (R2026b). Inputs: inDir/nm.json (tools/gen_nm.py) and inDir/nm4.json
% (tools/gen_nm4.py). Writes outDir/nm4.jsonl. Parts:
%   env      version, which() of the functions below under both path orders
%   sum3     M (a x b x c) built from rng(seed) (see gen_nm4.py); S3 = sum(M, 3), M3 = mean(M, 3);
%            head/tail = the first and last (up to) 4 words of M(:), to check the construction
%   sw       rng(seed); RW = nr calls of BCT 2019 randmio_und(W, iter). Then
%              S_bct = smallworldness_bu/wu (GraphVar helper) with BCT 2019 first on the path;
%              S_gv  = the same with GraphVar 2.03a's own src/ext/BCT first on the path;
%              eng   = cW, lW, cm, cp and S_eng from BCT 2019 with charpath(D, 0, 0) (diagonal and Inf excluded);
%            RW1..RWnr
%   swpa     small_world_propensity line for line with BCT 2019 distance_wei (option b), on asymmetric W:
%            every intermediate, including W_sym = symm_matrix's output; after = rand(1, 4)
%   shuffle  rng(seed); randM = random_shuffle(R, 1) (GraphVar helper); after = rand(1, 4)
p = jsondecode(fileread(fullfile(inDir, 'nm.json')));
q = jsondecode(fileread(fullfile(inDir, 'nm4.json')));
if ~isfolder(outDir), mkdir(outDir); end
out = fullfile(outDir, 'nm4.jsonl');
if isfile(out), delete(out); end
src = fullfile(gvDir, 'src');
gvBCT = fullfile(src, 'ext', 'BCT');
addpath(fullfile(src, 'ext', 'Network Community Toolbox'));
addpath(fullfile(src, 'gui', 'helper'));
addpath(bctDir);
swpDir = fullfile(outDir, 'swp_local');
nm_split_local(fullfile(src, 'ext', 'Network Community Toolbox', 'small_world_propensity.m'), swpDir);
addpath(swpDir, '-end');
fns = {'charpath', 'clustering_coef_bu', 'clustering_coef_wu', 'distance_bin', 'distance_wei', 'randmio_und', ...
       'smallworldness_bu', 'random_shuffle', 'symm_matrix'};

rec = newrec('nm4-env', 'env');
rec.version = version;
for f = fns, rec.notes.(['which_bct_' f{1}]) = clean(which(f{1}), bctDir, gvDir); end
addpath(gvBCT);
for f = fns, rec.notes.(['which_gv_' f{1}]) = clean(which(f{1}), bctDir, gvDir); end
rmpath(gvBCT);
emit(out, rec);

for c = cellify(q.sum3)
    c = c{1};
    rec = newrec(c.case, 'sum3');
    rec.a = c.a; rec.b = c.b; rec.c = c.c; rec.seed = c.seed; rec.kind = c.kind;
    try
        rng(c.seed, 'twister');
        U1 = rand(c.a, c.b, c.c);
        U2 = rand(c.a, c.b, c.c);
        M = (U1 - 0.5) .* 2.^(floor(64 * U2) - 32);
        clear U1 U2
        if strcmp(c.kind, 'negz'), M(1, 1, :) = -0; end
        k = min(4, numel(M));
        rec.outputs.head = M(1:k);
        rec.outputs.tail = M(end - k + 1:end);
        rec.outputs.S3 = sum(M, 3);
        rec.outputs.M3 = mean(M, 3);
    catch err
        rec.error = struct('identifier', err.identifier, 'message', err.message);
    end
    emit(out, rec);
    clear M
end

G = struct();
for g = cellify(p.graphs)
    g = g{1};
    G.(['nm_' key(g.name)]) = reshape(hex2num(cellstr(g.W)), g.n, g.n);
end
for g = cellify(q.graphs)
    g = g{1};
    G.(['nm4_' key(g.name)]) = reshape(hex2num(cellstr(g.W)), g.n, g.n);
end

for c = cellify(q.sw)
    c = c{1};
    rec = newrec(c.case, 'sw');
    rec.graph = c.graph; rec.src = c.src; rec.nr = c.nr; rec.iter = c.iter; rec.seed = c.seed;
    W = G.([c.src '_' key(c.graph)]);
    bin = all(W(:) == 0 | W(:) == 1);
    if bin, rec.which = 'bu'; else, rec.which = 'wu'; end
    try
        rng(c.seed, 'twister');
        RW = cell(1, c.nr);
        for t = 1:c.nr, RW{t} = randmio_und(W, c.iter); end
        rec.outputs.after = rand(1, 4);
        if bin, rec.outputs.S_bct = smallworldness_bu(W, RW); else, rec.outputs.S_bct = smallworldness_wu(W, RW); end
        cm = zeros(c.nr, 1); cp = zeros(c.nr, 1);
        for t = 1:c.nr, [cm(t), cp(t)] = eng_parts(RW{t}, bin); end
        [cW, lW] = eng_parts(W, bin);
        rec.outputs.cW = cW; rec.outputs.lW = lW; rec.outputs.cm = cm; rec.outputs.cp = cp;
        rec.outputs.S_eng = (cW / mean(cm)) / (lW / mean(cp));
        addpath(gvBCT);
        try
            if bin, rec.outputs.S_gv = smallworldness_bu(W, RW); else, rec.outputs.S_gv = smallworldness_wu(W, RW); end
        catch err
            rec.notes.gv_error = [err.identifier ': ' err.message];
        end
        rmpath(gvBCT);
        for t = 1:c.nr, rec.outputs.(sprintf('RW%d', t)) = RW{t}; end
    catch err
        rec.error = struct('identifier', err.identifier, 'message', err.message);
    end
    if any(strcmp(path_list(), gvBCT)), rmpath(gvBCT); end
    emit(out, rec);
end

for c = cellify(q.swpa)
    c = c{1};
    args = c.args;
    if isempty(args), args = {}; elseif ~iscell(args), args = cellstr(args); end
    rec = newrec(c.case, 'swpa');
    rec.graph = c.graph; rec.seed = c.seed; rec.args = args;
    W = G.(['nm4_' key(c.graph)]);
    try
        rng(c.seed, 'twister');
        [rec.outputs.SWP, rec.outputs.delta_C, rec.outputs.delta_L, parts] = swp(W, @nm_apsp_b, args{:});
        rec.outputs.after = rand(1, 4);
        f = fieldnames(parts);
        for i = 1:numel(f), rec.outputs.(f{i}) = parts.(f{i}); end
    catch err
        rec.error = struct('identifier', err.identifier, 'message', err.message);
    end
    emit(out, rec);
end

for c = cellify(q.shuffle)
    c = c{1};
    rec = newrec(c.case, 'shuffle');
    rec.n = c.n; rec.seed = c.seed;
    Rm = reshape(hex2num(cellstr(c.R)), c.n, c.n);
    try
        rng(c.seed, 'twister');
        rec.outputs.randM = random_shuffle(Rm, 1);
        rec.outputs.after = rand(1, 4);
    catch err
        rec.error = struct('identifier', err.identifier, 'message', err.message);
    end
    emit(out, rec);
end
end

function [cl, lam] = eng_parts(A, bin)
% BCT 2019 pieces of GraphVar's smallworldness helpers, with charpath(D, 0, 0): diagonal and Inf excluded
if bin
    cl = mean(clustering_coef_bu(A));
    D = distance_bin(A);
else
    cl = mean(clustering_coef_wu(A));
    E = find(A); A(E) = 1 ./ A(E);
    D = distance_wei(A);
end
lam = charpath(D, 0, 0);
end

function L = path_list()
L = strsplit(path, pathsep);
end

function [SWP, delta_C, delta_L, s] = swp(A, apsp, varargin)
% GraphVar 2.03a's small_world_propensity (Network Community Toolbox copy), line for line, with
% graphallshortestpaths replaced by apsp (as ref_nm2.m), also recording symm_matrix's output.
if isempty(varargin), varargin{1} = 'O'; end
s = struct();
SWP = NaN; delta_C = NaN; delta_L = NaN;
if sum(sum(A)) > 0
    bin_matrix = 0;
    if strcmp(varargin{1}, 'bin') == 1
        bin_matrix = 1;
        A = A > 0;
    end
    symcheck = abs(A - A');
    s.symcheck_sum = sum(sum(symcheck));
    if sum(sum(symcheck)) > 0
        W = symm_matrix(A, bin_matrix);
    else
        W = A;
    end
    s.W_sym = W;
    n = length(W);
    numb_connections = length(find(W > 0));
    avg_deg_unw = numb_connections / n;
    avg_rad_unw = avg_deg_unw / 2;
    avg_rad_eff = ceil(avg_rad_unw);
    W_reg = regular_matrix_generator(W, avg_rad_eff);
    W_rand = randomize_matrix(W);
    reg_path = avg_path(1 ./ W_reg, apsp);
    rand_path = avg_path(1 ./ W_rand, apsp);
    net_path = avg_path(1 ./ W, apsp);
    A = (net_path - rand_path);
    if A < 0
        A = 0;
    end
    diff_path = A / (reg_path - rand_path);
    if net_path == Inf || rand_path == Inf || reg_path == Inf
        diff_path = 1;
    end
    if diff_path > 1
        diff_path = 1;
    end
    reg_clus = avg_clus_matrix(W_reg, varargin{1});
    rand_clus = avg_clus_matrix(W_rand, varargin{1});
    net_clus = avg_clus_matrix(W, varargin{1});
    B = (reg_clus - net_clus);
    if B < 0
        B = 0;
    end
    diff_clus = B / (reg_clus - rand_clus);
    if isnan(reg_clus) || isnan(rand_clus) || isnan(net_clus)
        diff_clus = 1;
    end
    if diff_clus > 1
        diff_clus = 1;
    end
    SWP = 1 - (sqrt(diff_clus^2 + diff_path^2) / sqrt(2));
    delta_C = diff_clus;
    delta_L = diff_path;
    s.avg_rad_eff = avg_rad_eff; s.W_reg = W_reg; s.W_rand = W_rand;
    s.reg_path = reg_path; s.rand_path = rand_path; s.net_path = net_path;
    s.reg_clus = reg_clus; s.rand_clus = rand_clus; s.net_clus = net_clus;
    s.C_net = clustering_coef_matrix(W, varargin{1});
end
end

function Len = avg_path(M, apsp)
n = length(M);
M = sparse(M);
D = apsp(M);
for i = 1:n
    for j = 1:n
        if isinf(D(i, j)) == 1
            D(i, j) = 0;
        end
    end
end
Len = mean(mean(D));
end

function s = clean(w, bctDir, gvDir)
s = strrep(strrep(w, bctDir, '<bct>'), gvDir, '<gv>');
end

function k = key(name)
k = strrep(strrep(name, '-', '_'), '.', 'p');
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
    if ischar(v) || isstruct(v), continue, end
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
