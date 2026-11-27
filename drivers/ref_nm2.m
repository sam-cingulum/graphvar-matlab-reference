function ref_nm2(bctDir, gvDir, inDir, outDir)
% Second null-model battery (R2026b). Inputs: inDir/nm.json (tools/gen_nm.py) and inDir/nm2.json
% (tools/gen_nm2.py). Writes outDir/nm2.jsonl. Parts:
%   state       the twister State after rng(1, 'twister'), and its last word after one rand
%   words       crafted-stream check: which index word makes rand return the crafted doubles
%   craft       randperm(n, k) on a crafted stream; after = rand(1, 4)
%   seeds       randperm(n, k) after rng(seed, 'twister'); after = rand(1, 4)
%   swp         for every small_world_propensity call in nm.json, with graphallshortestpaths replaced by
%               option a (MATLAB distances(graph(L))) and option b (BCT 2019 distance_wei(L)): SWP,
%               delta_C, delta_L, every intermediate, and D for the network itself under both options.
%               On a release that still has graphallshortestpaths, also option g (graphallshortestpaths
%               itself, through the same line-for-line code) and part swp_orig (GraphVar's own function).
%   env         version, exist() of graphallshortestpaths, distances and graph
p = jsondecode(fileread(fullfile(inDir, 'nm.json')));
q = jsondecode(fileread(fullfile(inDir, 'nm2.json')));
if ~isfolder(outDir), mkdir(outDir); end
out = fullfile(outDir, 'nm2.jsonl');
if isfile(out), delete(out); end
src = fullfile(gvDir, 'src');
addpath(fullfile(src, 'ext', 'Network Community Toolbox'));
addpath(fullfile(src, 'gui', 'helper'));
addpath(bctDir);
swpDir = fullfile(outDir, 'swp_local');
nm_split_local(fullfile(src, 'ext', 'Network Community Toolbox', 'small_world_propensity.m'), swpDir);
addpath(swpDir, '-end');

rec = newrec('nm2-env', 'env');
rec.version = version;
rec.exist_graphallshortestpaths = exist('graphallshortestpaths'); %#ok<EXIST>
rec.exist_distances = exist('distances'); %#ok<EXIST>
emit(out, rec);
rng(1, 'twister');
st = rng;
rec = newrec('nm2-state', 'state');
rec.outputs.state_size = size(st.State);
rec.outputs.state = double(st.State(:))';
rand;
st2 = rng;
rec.outputs.last_after_one_rand = double(st2.State(end));
emit(out, rec);

% crafted stream: find the index word that makes rand return the crafted doubles
w = q.words(1);
useIdx = NaN;
rec = newrec('nm2-words', 'words');
for idx = [0 1 624 625]
    set_state(w.state, idx);
    u = rand(1, numel(w.u));
    rec.outputs.(sprintf('rand_idx%d', idx)) = u;
    if isnan(useIdx) && isequal(u(:), w.u(:)), useIdx = idx; end
end
rec.outputs.use_idx = useIdx;
emit(out, rec);
for c = cellify(q.craft)
    c = c{1};
    rec = newrec(c.case, 'craft');
    rec.n = c.n; rec.k = c.k; rec.u = c.u(:)';
    if isnan(useIdx)
        rec.error = struct('identifier', 'nm2:craft', 'message', 'no index word reproduces the crafted doubles');
    else
        set_state(c.state, useIdx);
        rec.outputs.check = rand(1, 1);
        set_state(c.state, useIdx);
        rec.outputs.p = randperm(c.n, c.k);
        rec.outputs.after = rand(1, 4);
    end
    emit(out, rec);
end
for c = cellify(q.seeds)
    c = c{1};
    rec = newrec(sprintf('nm2-seed-n%d-k%d-s%d', c.n, c.k, c.seed), 'seeds');
    rec.n = c.n; rec.k = c.k; rec.seed = c.seed;
    rng(c.seed, 'twister');
    rec.outputs.p = randperm(c.n, c.k);
    rec.outputs.after = rand(1, 4);
    emit(out, rec);
end

G = struct();
for g = cellify(p.graphs)
    g = g{1};
    G.(key(g.name)) = reshape(hex2num(cellstr(g.W)), g.n, g.n);
end
for c = cellify(p.calls)
    c = c{1};
    if ~strcmp(c.fn, 'small_world_propensity'), continue, end
    args = c.args;
    if isempty(args), args = {}; elseif ~iscell(args), args = num2cell(args(:)'); end
    W = G.(key(c.graph));
    opts = {'a', 'b'};
    if exist('graphallshortestpaths') %#ok<EXIST>
        opts{end + 1} = 'g';
        rec = newrec(sprintf('%s-orig', c.case), 'swp_orig');
        rec.call = c.case; rec.graph = c.graph; rec.seed = c.seed; rec.args = args;
        try
            rng(c.seed, 'twister');
            [rec.outputs.SWP, rec.outputs.delta_C, rec.outputs.delta_L] = small_world_propensity(W, args{:});
            rec.outputs.after = rand(1, 4);
        catch err
            rec.error = struct('identifier', err.identifier, 'message', err.message);
        end
        emit(out, rec);
    end
    for opt = opts
        rec = newrec(sprintf('%s-opt%s', c.case, opt{1}), 'swp');
        rec.call = c.case; rec.graph = c.graph; rec.seed = c.seed; rec.args = args; rec.option = opt{1};
        if strcmp(opt{1}, 'g'), apsp = @(M) graphallshortestpaths(M); else, apsp = str2func(['nm_apsp_' opt{1}]); end
        try
            rng(c.seed, 'twister');
            [rec.outputs.SWP, rec.outputs.delta_C, rec.outputs.delta_L, parts] = swp(W, apsp, args{:});
            rec.outputs.after = rand(1, 4);
            f = fieldnames(parts);
            for i = 1:numel(f), rec.outputs.(f{i}) = parts.(f{i}); end
            A = W;
            if ~isempty(args) && strcmp(args{1}, 'bin'), A = double(W > 0); end
            rec.outputs.D_net = apsp(sparse(1 ./ A));
        catch err
            rec.error = struct('identifier', err.identifier, 'message', err.message);
        end
        emit(out, rec);
    end
end
end

function [SWP, delta_C, delta_L, s] = swp(A, apsp, varargin)
% GraphVar 2.03a's small_world_propensity (Network Community Toolbox copy), line for line, with
% graphallshortestpaths replaced by apsp. Local functions come from the split files.
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
    if sum(sum(symcheck)) > 0
        W = symm_matrix(A, bin_matrix);
    else
        W = A;
    end
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
% GraphVar's avg_path_matrix with graphallshortestpaths replaced by apsp
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

function set_state(words, idx)
rng(1, 'twister');
st = rng;
S = st.State;
S(1:numel(words)) = uint32(words(:));
S(end) = uint32(idx);
st.State = S;
rng(st);
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
